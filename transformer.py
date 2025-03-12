import torch
from torch import nn
import numpy as np
from dataset import TextDataset
from typing import Tuple, List
import torch.nn.functional as F
from tqdm import tqdm

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * np.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor) -> torch.Tensor:
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.shape[1]])

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int):
        super(LearnedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(num_embeddings=maxlen, embedding_dim=emb_size)
    
    def forward(self, token_embedding: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        return self.dropout(token_embedding + self.embedding(torch.arange(0, token_embedding.shape[1], device=device)))

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int, dropout: float, padding_idx: int, maxlen: int = 100):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_size, padding_idx=padding_idx)
        self.pos_embedding = PositionalEncoding(emb_size=emb_size, dropout=dropout, maxlen=maxlen)
        self.emb_size = emb_size

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.pos_embedding(self.embedding(tokens.long()) * np.sqrt(self.emb_size))

class GPT(nn.Module):
    def __init__(self,
                 dataset: TextDataset,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(GPT, self).__init__()
        self.dataset = dataset
        self.de_vocab_size = dataset.de_vocab_size
        self.en_vocab_size = dataset.en_vocab_size
        self.de_embedding = TokenEmbedding(self.de_vocab_size, emb_size, dropout, padding_idx=self.dataset.de_pad_id)
        self.en_embedding = TokenEmbedding(self.en_vocab_size, emb_size, dropout, padding_idx=self.dataset.en_pad_id)
        self.transformer = nn.Transformer(d_model=emb_size,
                                          nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout,
                                          batch_first=True)
        self.head = nn.Linear(emb_size, self.en_vocab_size)

    def no_future(self, L: int) -> torch.Tensor:
        device = next(self.parameters()).device
        return torch.triu(torch.ones((L, L), dtype=bool, device=device), diagonal=1)

    def forward(self, de_indices: torch.Tensor, en_indices: torch.Tensor) -> torch.Tensor:
        de_emb = self.de_embedding(de_indices)
        en_emb = self.en_embedding(en_indices)
        decoder_mask, de_pad_mask, en_pad_mask = self.no_future(en_indices.shape[1]), (de_indices == self.dataset.de_pad_id), \
                                                 (en_indices == self.dataset.en_pad_id)
        out = self.transformer(src=de_emb, tgt=en_emb, tgt_mask=decoder_mask, src_key_padding_mask=de_pad_mask, 
                                tgt_key_padding_mask=en_pad_mask, memory_key_padding_mask=de_pad_mask)
        return self.head(out)
    
    def encode(self, de_indices: torch.Tensor) -> torch.Tensor:
        de_emb = self.de_embedding(de_indices)
        return self.transformer.encoder(de_emb, src_key_padding_mask=(de_indices == self.dataset.de_pad_id))

    def decode(self, de_indices: torch.Tensor, en_indices: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        en_emb = self.en_embedding(en_indices)
        return self.transformer.decoder(en_emb, memory=h, tgt_mask=self.no_future(en_indices.shape[1]), 
                                        tgt_key_padding_mask=(en_indices == self.dataset.en_pad_id), 
                                        memory_key_padding_mask=(de_indices == self.dataset.de_pad_id))
    
    @torch.inference_mode()
    def inference(self, de_indices: torch.Tensor, beam_size: int = 0, lmbda: float = 0, return_indices: bool = False) -> List[str]:
        self.eval()
        B = de_indices.shape[0]
        device = next(self.parameters()).device
        h = self.encode(de_indices)

        if beam_size == 0:
            en_indices = torch.full((B, 1), self.dataset.en_bos_id, device=device)

            for i in range(self.dataset.max_length - 2):
                out = self.decode(de_indices, en_indices, h)[:, -1]
                logits = self.head(out)
                new_indices = logits.argmax(axis=1).unsqueeze(dim=-1)
                en_indices = torch.cat((en_indices, new_indices), dim=1)
            
            en_indices = torch.cat((en_indices, torch.full((B, 1), self.dataset.en_eos_id, device=device)), dim=1)
        else:
            seqs = torch.stack([torch.full((B, 1), self.dataset.en_bos_id, device=device)])
            probs = torch.stack([torch.full((B,), 0, device=device)])
            best_seqs = []
            best_probs = []
            for i in range(self.dataset.max_length - 2):
                print("BS: ", i)
                nseqs = []
                nprobs = []
                for seq, prob in zip(seqs, probs):
                    out = self.decode(de_indices, seq, h)[:, -1]
                    logits = F.log_softmax(self.head(out), dim=1)
                    p_values, indices = torch.topk(logits, k=beam_size, dim=1)
                    for new_p, new_indices in zip(p_values.T, indices.T):
                        nseqs.append(torch.cat((seq, new_indices.unsqueeze(dim=1)), dim=1))
                        nprobs.append(prob + new_p + lmbda)
                nseqs = torch.stack(nseqs)
                nprobs = torch.stack(nprobs)

                fin_probs = nprobs.clone()
                fin_probs[nseqs[:, :, -1] != self.dataset.en_eos_id] = float('-inf')
                vals, ids = fin_probs.max(dim=0)
                best_probs.append(vals)
                best_seqs.append(torch.cat((nseqs[ids, torch.arange(B)], torch.full((B, self.dataset.max_length - 1 - nseqs.shape[2]), self.dataset.en_pad_id, device=device)), dim=1))

                nprobs[nseqs[:, :, -1] == self.dataset.en_eos_id] = float('-inf')
                nprobs, sort_ids = torch.topk(nprobs, beam_size, dim=0)
                probs = nprobs
                seqs = nseqs[sort_ids, torch.arange(nseqs.shape[1]).unsqueeze(0)]

            best_probs.reverse()
            best_seqs.reverse()
            vals, ids = torch.stack(best_probs).max(dim=0)
            en_indices = torch.stack(best_seqs)[ids, torch.arange(B)]
            en_indices = torch.cat((en_indices, torch.full((B, 1), self.dataset.en_eos_id, device=device)), dim=1)

        first_eos = (en_indices == self.dataset.en_eos_id).int().argmax(axis=1)
        for i in range(B):
            en_indices[i, first_eos[i] + 1:] = self.dataset.en_pad_id
        print("FINISHED INFERECE")
        if return_indices:
            return self.dataset.ids2text(en_indices, 'en'), en_indices
        else:
            return self.dataset.ids2text(en_indices, 'en')

    @torch.inference_mode()
    def estimate_likelyhood(self, de_indices: torch.Tensor, en_indices: torch.tensor):
        first_eos = (en_indices == self.dataset.en_eos_id).int().argmax(axis=1)
        B = de_indices.shape[0]
        device = next(self.parameters()).device
        h = self.encode(de_indices)
        out = self.decode(de_indices, en_indices[:, :-1], h)
        logits = F.log_softmax(self.head(out), dim=2)
        probs = []
        for i in range(B):
            probs.append(logits[i][torch.arange(first_eos[i]), en_indices[i, 1:first_eos[i] + 1]].sum().item())
        return np.sum(probs)
