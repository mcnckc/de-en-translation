import copy
import torch
from torch import Tensor
from torch import nn
from typing import Tuple, Union, Callable, Optional, List
from dataset import TextDataset
from transformer import GPT, TokenEmbedding
import torch.nn.functional as F

class GPT_ILDUS(GPT):
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
        self.transformer = GPT_ILDUSTransformer(d_model=emb_size,
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
        decoder_mask, de_pad_mask, en_pad_mask = self.no_future(en_indices.shape[1]), (de_indices == self.dataset.de_pad_id), \
                                                 (en_indices == self.dataset.en_pad_id)
        h = self.transformer.encoder(de_emb, src_key_padding_mask=de_pad_mask)
        
        out = self.transformer.decoder(en_emb, memory=h, tgt_mask=self.no_future(en_indices.shape[1]), 
                                        tgt_key_padding_mask=en_pad_mask, memory_key_padding_mask=de_pad_mask)

        attn = self.transformer.decoder(en_emb, memory=h, tgt_key_padding_mask=en_pad_mask, memory_key_padding_mask=de_pad_mask, attn_mode=True)
        return self.head(out), attn
    
    @torch.inference_mode()
    def inference(self, de_indices: torch.Tensor, de_lengths: torch.Tensor, de_texts: List[str], beam_size: int = 0, lmbda: float = 0, return_indices: bool = False) -> List[str]:
        en_texts, en_indices = super().inference(de_indices, beam_size, lmbda, return_indices=True)
        de_words = [text.split() for text in de_texts]
        en_words = [text.split() for text in en_texts]
        en_lengths = (en_indices == self.dataset.en_eos_id).int().argmax(axis=1) + 1
        en_indices = en_indices[:, :en_lengths.max()]
        _, attn = self.forward(de_indices, en_indices[:, :-1])
        B = de_indices.shape[0]
        for i in range(B):
            N, M = len(en_words[i]), len(de_words[i])
            if M != de_lengths[i] - 2:
                raise AssertionError('Wrong de tokenization in inference')
            if N != en_lengths[i] - 2:
                raise AssertionError('Wrong en tokenization in inference')
            unk_ids = (en_indices[i] == self.dataset.en_unk_id).nonzero()
            aligns = attn[i][:N, 1:M + 1].argmax(dim=1)
            for unk_id in unk_ids:
                unk_id = unk_id.item()
                if aligns[unk_id - 1] >= M:
                    raise AssertionError('Aligned wrong')
                de_word = de_words[i][aligns[unk_id - 1]]
                translation = self.dataset.dictionary.get(de_word, de_word)
                en_words[i][unk_id - 1] = translation
        if return_indices:
            return [' '.join(words) for words in en_words], en_indices
        else:
            return [' '.join(words) for words in en_words]



class GPT_ILDUSTransformer(nn.Transformer):
    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        nn.Module.__init__(self)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                activation, layer_norm_eps, batch_first, norm_first,
                                                **factory_kwargs)
        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                activation, layer_norm_eps, batch_first, norm_first,
                                                **factory_kwargs)
        alignment_layer = PenultimateLayer(d_model, nhead, dim_feedforward, dropout,
                                           activation, layer_norm_eps, batch_first, norm_first,
                                           **factory_kwargs)
        decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)

        self.decoder = GPT_ILDUSDecoder(decoder_layer, alignment_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.batch_first = batch_first


class GPT_ILDUSDecoder(nn.TransformerDecoder):
    def __init__(self, decoder_layer, alignment_layer, num_layers, norm=None):
        nn.Module.__init__(self)
        self.layers = _get_clones(decoder_layer, num_layers - 1)
        self.layers.insert(num_layers - 2, alignment_layer)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None, attn_mode: bool = False) -> Tensor:
        output = tgt

        for i in range(self.num_layers - 1):
            output = self.layers[i](output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        output, attn = output
        if attn_mode:
            return attn

        output = self.layers[-1](output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)

        return output

class PenultimateLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        super(PenultimateLayer, self).__init__(d_model, nhead, dim_feedforward, dropout,
                 activation, layer_norm_eps, batch_first, norm_first, device, dtype)
        self.alignment_head = torch.randint(nhead, (1,)).item()

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        x = tgt
        x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
        nx, attn = self._mha_block(x, memory, memory_mask, memory_key_padding_mask)
        x = self.norm2(x + nx)
        x = self.norm3(x + self._ff_block(x))

        return x, attn[:, self.alignment_head]

    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x, attn = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=True, average_attn_weights=False)
        return self.dropout2(x), attn

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])