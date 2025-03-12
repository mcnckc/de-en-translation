import copy
import torch
from torch import Tensor
from torch import nn
from typing import Tuple, Union, Callable, Optional
from dataset import TextDataset
from transformer import GPT, TokenEmbedding
import torch.nn.functional as F

class Aligner(GPT):
    def __init__(self,
                 dataset: TextDataset,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        nn.Module.__init__(self)
        self.dataset = dataset
        self.de_vocab_size = dataset.de_vocab_size
        self.en_vocab_size = dataset.en_vocab_size
        self.de_embedding = TokenEmbedding(self.de_vocab_size, emb_size, dropout, padding_idx=self.dataset.de_pad_id)
        self.en_embedding = TokenEmbedding(self.en_vocab_size, emb_size, dropout, padding_idx=self.dataset.en_pad_id)
        self.transformer = AlignerTransformer(d_model=emb_size,
                                          nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout,
                                          batch_first=True)
        self.head = nn.Linear(emb_size, self.en_vocab_size)
    
    def forward(self, de_indices: torch.Tensor, en_indices: torch.Tensor) -> torch.Tensor:
        de_emb = self.de_embedding(de_indices)
        en_emb = self.en_embedding(en_indices)
        de_pad_mask, en_pad_mask = (de_indices == self.dataset.de_pad_id), (en_indices == self.dataset.en_pad_id)
        out, attentions = self.transformer(src=de_emb, tgt=en_emb, src_key_padding_mask=de_pad_mask, 
                                tgt_key_padding_mask=en_pad_mask, memory_key_padding_mask=de_pad_mask)
        return self.head(out), attentions
    


class AlignerTransformer(nn.Transformer):
    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        nn.Module.__init__(self)
        factory_kwargs = {'device': device, 'dtype': dtype}

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                activation, layer_norm_eps, batch_first, norm_first,
                                                **factory_kwargs)
        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

       
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                activation, layer_norm_eps, batch_first, norm_first,
                                                **factory_kwargs)
        alignment_layer = AlignmentLayer(d_model, nhead, dim_feedforward, dropout,
                                         activation, layer_norm_eps, batch_first, norm_first,
                                         **factory_kwargs)
        self.decoder = AlignerDecoder(decoder_layer, alignment_layer, num_decoder_layers)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.batch_first = batch_first

class AlignerDecoder(nn.TransformerDecoder):
     def __init__(self, decoder_layer, alignment_layer, num_layers):
        nn.Module.__init__(self)
        self.layers = _get_clones(decoder_layer, num_layers - 1)
        self.layers.append(alignment_layer)
        self.num_layers = num_layers
        self.norm = None

class AlignmentLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        nn.Module.__init__(self)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        self.multihead_attn = SeparateMultiHeadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                 **factory_kwargs)
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation
        
    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        x = tgt
        x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
        return self.multihead_attn(x, memory, memory,
                                attn_mask=memory_mask,
                                key_padding_mask=memory_key_padding_mask,
                                need_weights=True)
    def unfreeze_weights(self):
        self.multihead_attn.in_proj_weight.requires_grad = True
        self.multihead_attn.in_proj_bias.requires_grad = True

class SeparateMultiHeadAttention(nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        nn.Module.__init__(self)
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = False # we need to separate different attention weigths for fine-tuning

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj_weight = nn.parameter.Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
        self.k_proj_weight = nn.parameter.Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
        self.v_proj_weight = nn.parameter.Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
        self.register_parameter('in_proj_weight', None)

        if bias:
            self.in_proj_bias = nn.parameter.Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.modules.linear.NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        if add_bias_kv:
            self.bias_k = nn.parameter.Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = nn.parameter.Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])