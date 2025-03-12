import os
import math
import torch
import numpy as np
from typing import Union, List, Tuple, Iterator
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor
from torch.utils.data import Dataset, Sampler
from adjustable_constants import DATASET_FOLDER, ADDITIONAL_DATA_FOLDER

class TextDataset(Dataset):
    ADDITIONAL_DATA_FOLDER = ADDITIONAL_DATA_FOLDER #folder to contain aligner artifacts
    NO_ALIGN_INDEX = 179179 #special index to be ignored in alignment cross-entropy loss, do not change

    def __init__(self, train: bool = True, sp_model_prefix: Tuple[str, str] = None, sub_sample: float = 1,
                 vocab_size: Tuple[int, int] = (2000, 2000), normalization_rule_name: str = 'nmt_nfkc_cf',
                 model_type: Tuple[str, str] = ('bpe', 'bpe'), max_length: int = 128, reverse_text: bool = False,
                 full_vocab: Tuple[bool, bool] = (False, False), mode: str = 'default', shuffle: bool = False, 
                 enable_dictionary: bool = False, force_training: bool = True):

        file_prefix = DATASET_FOLDER + 'train.de-en' if train else DATASET_FOLDER + 'val.de-en'
        lang_prefs = ('de', 'en')
        models = []
        if not os.path.exists('tokenizers'):
            os.mkdir('tokenizers')
        for i in range(2):
            if force_training or not os.path.isfile('tokenizers/' + lang_prefs[i] + '-' + sp_model_prefix[i] + '.model'):
                SentencePieceTrainer.train(
                    input=DATASET_FOLDER + 'train.de-en.' + lang_prefs[i], vocab_size=vocab_size[i],
                    model_type=model_type[i], model_prefix='tokenizers/' + lang_prefs[i] + '-' + sp_model_prefix[i],
                    normalization_rule_name=normalization_rule_name, pad_id=4, use_all_vocab=full_vocab[i], split_by_unicode_script=False,
                    user_defined_symbols='<MASK>', minloglevel=1
                )
            models.append(SentencePieceProcessor(model_file='tokenizers/' + lang_prefs[i] + '-' + sp_model_prefix[i] + '.model'))


        self.de_sp_model, self.en_sp_model = models[0], models[1]
        
        with open(file_prefix + '.de', encoding='utf-8') as file:
            de_texts = file.readlines()
        with open(file_prefix + '.en', encoding='utf-8') as file:
            en_texts = file.readlines()

        self.train = train
        self.mode = mode
        self.reverse_text = reverse_text
        self.sub_sample = math.ceil(sub_sample * len(de_texts))
        if shuffle:
            ids = torch.randperm(len(de_texts))[:self.sub_sample]
            self.de_texts = [de_texts[i.item()] for i in ids]
            self.en_texts = [en_texts[i.item()] for i in ids]
        else:
            self.de_texts = de_texts[:self.sub_sample]
            self.en_texts = en_texts[:self.sub_sample]

        self.de_indices = self.my_encode(self.de_sp_model, self.de_texts)
        self.en_indices = self.my_encode(self.en_sp_model, self.en_texts)

        self.de_pad_id, self.de_unk_id, self.de_bos_id, self.de_eos_id = \
            self.de_sp_model.pad_id(), self.de_sp_model.unk_id(), \
            self.de_sp_model.bos_id(), self.de_sp_model.eos_id()
        
        self.en_pad_id, self.en_unk_id, self.en_bos_id, self.en_eos_id = \
            self.en_sp_model.pad_id(), self.en_sp_model.unk_id(), \
            self.en_sp_model.bos_id(), self.en_sp_model.eos_id()
        
        self.en_mask_id = self.en_sp_model.PieceToId('<MASK>')
        self.max_length = max_length
        self.de_vocab_size = self.de_sp_model.vocab_size()
        self.en_vocab_size = self.en_sp_model.vocab_size()   

        if self.mode == 'aligns' and self.train:
            self.attns = torch.load(self.ADDITIONAL_DATA_FOLDER + 'training_alignments.pt', map_location=torch.device('cpu'))
        if enable_dictionary:
            self.dictionary = torch.load(self.ADDITIONAL_DATA_FOLDER + 'de-en-dictionary.pt')

    def my_encode(self, sp_model, texts: Union[str, List[str]]):
        if type(texts) is str:
            texts = [texts]
        tokens = []
        for text in texts:
            cur_tokens = []
            for word in text.split():
                t = sp_model.encode(word)
                cur_tokens.append(t[0] if len(t) > 0 else sp_model.unk_id())
            tokens.append(cur_tokens)
        return tokens[0] if len(tokens) == 1 else tokens

    def text2ids(self, texts: Union[str, List[str]], lang: str) -> Union[List[int], List[List[int]]]:
        return self.my_encode(self.de_sp_model, texts) if lang == 'de' else self.my_encode(self.en_sp_model, texts)

    def ids2text(self, ids: Union[torch.Tensor, List[int], List[List[int]]], lang: str) -> Union[str, List[str]]:
        if torch.is_tensor(ids):
            assert len(ids.shape) <= 2, 'Expected tensor of shape (length, ) or (batch_size, length)'
            ids = ids.cpu().tolist()

        return self.de_sp_model.decode(ids) if lang == 'de' else self.en_sp_model.decode(ids)

    def __len__(self):
        return len(self.de_indices)

    def __getitem__(self, item: int):
        de_ids = self.de_indices[item][:min(len(self.de_indices[item]), self.max_length - 2)]
        en_ids = self.en_indices[item][:min(len(self.en_indices[item]), self.max_length - 2)]

        if self.reverse_text:
            de_ids.reverse()

        de_indices = [self.de_bos_id] + de_ids + [self.de_eos_id] + [self.de_pad_id] * (self.max_length - len(de_ids) - 2)
        
        en_indices = [self.en_bos_id] + en_ids + [self.en_eos_id] + [self.en_pad_id] * (self.max_length - len(en_ids) - 2)
        
            
        if self.mode == 'default':
            if self.train:
                return torch.tensor(de_indices), torch.tensor(en_indices), len(de_ids) + 2, len(en_ids) + 2
            else:
                return torch.tensor(de_indices), torch.tensor(en_indices), len(de_ids) + 2, len(en_ids) + 2, self.en_texts[item]
        elif self.mode == 'both_texts':
            return torch.tensor(de_indices), torch.tensor(en_indices), len(de_ids) + 2, len(en_ids) + 2, self.de_texts[item], self.en_texts[item]
        elif self.mode == 'aligns':
            if self.train:
                attn = self.attns[item][:min(len(self.attns[item]), self.max_length - 1)]
                lpad = torch.full((1, ), self.NO_ALIGN_INDEX, dtype=int)
                rpad = torch.full((self.max_length - len(attn) - 1, ), self.NO_ALIGN_INDEX, dtype=int)
                attn[attn != self.NO_ALIGN_INDEX] += 1
                attn = torch.cat((lpad, attn, rpad))
                return torch.tensor(de_indices), torch.tensor(en_indices), len(de_ids) + 2, len(en_ids) + 2, attn
            else:
                return torch.tensor(de_indices), torch.tensor(en_indices), len(de_ids) + 2, len(en_ids) + 2, self.de_texts[item], self.en_texts[item]
        else:
            raise AssertionError('Wrong dataset mode')

def get_datasets(model_types: Tuple[str, str], sub_sample: float = 0.3, vocab_sizes: Tuple[int, int] = (2000, 2000), 
                 full_vocabs: Tuple[bool, bool] = (False, False), max_length: int = 128, reverse_text: bool = False, mode: str = 'default', 
                 enable_dictionary: bool = False, force_training: bool = True):
    prefs = (model_types[0] + '-' + str(max_length) + '-' + str(vocab_sizes[0]), model_types[1] + '-' + str(max_length) + '-' + str(vocab_sizes[1]))
    return TextDataset(train=True, sub_sample=sub_sample, vocab_size=vocab_sizes, sp_model_prefix=prefs, model_type=model_types, 
                       max_length=max_length, reverse_text=reverse_text, full_vocab=full_vocabs, mode=mode, enable_dictionary=enable_dictionary,
                       force_training=force_training), \
            TextDataset(train=False, vocab_size=vocab_sizes, sp_model_prefix=prefs, model_type=model_types, 
                        max_length=max_length, reverse_text=reverse_text, full_vocab=full_vocabs, mode=mode, force_training=force_training)

class TestDataset(Dataset):
    def __init__(self, val_set: TextDataset):
        self.reverse_text = val_set.reverse_text
        self.de_sp_model = val_set.de_sp_model
        with open(DATASET_FOLDER + 'test.de-en.de', encoding='utf-8') as file:
            self.de_texts = file.readlines()

        self.de_indices = self.my_encode(self.de_texts)

        self.de_pad_id, self.de_unk_id, self.de_bos_id, self.de_eos_id = \
            self.de_sp_model.pad_id(), self.de_sp_model.unk_id(), \
            self.de_sp_model.bos_id(), self.de_sp_model.eos_id()
        
        self.max_length = val_set.max_length

    def my_encode(self, texts: Union[str, List[str]]):
        if type(texts) is str:
            texts = [texts]
        tokens = []
        for text in texts:
            cur_tokens = []
            for word in text.split():
                t = self.de_sp_model.encode(word)
                cur_tokens.append(t[0] if len(t) > 0 else self.de_sp_model.unk_id())
            tokens.append(cur_tokens)
        return tokens[0] if len(tokens) == 1 else tokens

    def __len__(self):
        return len(self.de_indices)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, int]:

        de_ids = self.de_indices[item][:min(len(self.de_indices[item]), self.max_length - 2)]

        if self.reverse_text:
            de_ids.reverse()

        de_indices = [self.de_bos_id] + de_ids + [self.de_eos_id] + [self.de_pad_id] * (self.max_length - len(de_ids) - 2)
        return torch.tensor(de_indices), len(de_ids) + 2, self.de_texts[item]

class LengthAwareSampler(Sampler[int]):

    def __init__(self, dataset: TextDataset, nbins: int, batch_size: int):
        self.lengths = np.array([len(ids) for ids in dataset.de_indices])
        self.N = len(self.lengths)
        self.nbins = nbins
        self.batch_size = batch_size
    
    def __iter__(self) -> Iterator[int]:
        ids = np.random.permutation(self.N)
        arrays = np.array_split(ids, self.nbins)
        arrays = [a[np.argsort(self.lengths[a])] for a in arrays]
        arrays = np.array_split(np.concatenate(arrays), np.arange(self.batch_size, self.N, self.batch_size))
        return iter(np.concatenate([arrays[i] for i in np.random.permutation(len(arrays))]))
    
    def __len__(self) -> int:
        return self.N

class LengthAwareSampler2(Sampler[int]):

    def __init__(self, dataset: TextDataset, batch_size: int):
        self.sorted_ids = np.argsort([len(ids) for ids in dataset.de_indices])
        self.N = len(self.sorted_ids)
        self.batch_size = batch_size
    
    def __iter__(self) -> Iterator[int]:
        sorted_ids = self.sorted_ids
        batches = []
        while len(sorted_ids) > 0:
            cur_batch_size = min(self.batch_size, len(sorted_ids))
            l = np.random.randint(len(sorted_ids) - cur_batch_size + 1)
            batches.append(sorted_ids[l:l + cur_batch_size])
            sorted_ids = np.delete(sorted_ids, np.s_[l:l + cur_batch_size])
        return iter(np.concatenate(batches))
    
    def __len__(self) -> int:
        return self.N