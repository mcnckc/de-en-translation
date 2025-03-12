import os
import torch
import numpy as np
from dataset import TextDataset
from aligner import Aligner
from torch.utils.data import DataLoader
#import matplotlib.pyplot as plt
from tqdm import tqdm

@torch.inference_mode()
def employ_aligner(device, folder: str, num_workers: int):
    train_set = TextDataset(train=True, sp_model_prefix=('tokenizer_for_aligner_employment', 'tokenizer_for_aligner_employment'), 
                               sub_sample=1, vocab_size=(20000, 10000), model_type=('word', 'word'), max_length=82,
                               full_vocab=(True, False), mode='both_texts', force_training=True)
    model = Aligner(train_set, num_encoder_layers=3, num_decoder_layers=3, emb_size=192, nhead=6, dim_feedforward=384, dropout=0.055).to(device)
    checkpoint = torch.load(folder + '/aligner_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    if not os.path.exists(TextDataset.ADDITIONAL_DATA_FOLDER):
        os.mkdir(TextDataset.ADDITIONAL_DATA_FOLDER)
    build_dictionary(train_set, model, num_workers)
    save_alignments(train_set, model, num_workers)

@torch.inference_mode()
def get_attention(model, de_indices, en_indices, de_lengths, en_lengths, swap_langs: bool = False, view_mode: bool = False):
    if not swap_langs:
        source, target, source_lengths, target_lengths = de_indices, en_indices, de_lengths, en_lengths
    else:
        source, target, source_lengths, target_lengths = en_indices, de_indices, en_lengths, de_lengths
    device = next(model.parameters()).device 
    source = source[:, :source_lengths.max() - 1].to(device)
    target = target[:, 1:target_lengths.max() - 1].to(device)
    source[source == model.dataset.de_eos_id] = model.dataset.de_pad_id
    target[target == model.dataset.de_eos_id] = model.dataset.de_pad_id
    logits, attention = model(source, target)
    if view_mode:
        return logits, attention
    else:
        return attention

@torch.inference_mode()
def build_dictionary(train_set: TextDataset, de_en_model, num_workers: int):
    batch_size = 64
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    device = next(de_en_model.parameters()).device
    de_en_model.eval()
    dictionary = dict()
    attn_value = dict()
    for de_indices, en_indices, de_lengths, en_lengths, de_texts, en_texts in tqdm(train_loader, desc='Building dictionary'):
        de_indices, en_indices = de_indices.to(device), en_indices.to(device)
        de_words = [text.split() for text in de_texts]
        en_words = [text.split() for text in en_texts]
        B = de_indices.shape[0]
        attn = get_attention(de_en_model, de_indices, en_indices, de_lengths, en_lengths, False)
        
        for i in range(B):
            if len(de_words[i]) != de_lengths[i] - 2:
                raise AssertionError('Wrong de tokenization')
            if len(en_words[i]) != en_lengths[i] - 2:
                raise AssertionError('Wrong en tokenization')
            N, M = len(en_words[i]), len(de_words[i])
            en_aligns = attn[i][:N - 1, 1:M].argmax(dim=1)
            de_aligns = attn[i][:N - 1, 1:M].argmax(dim=0)
            en_mask, de_mask = torch.zeros(N - 1, M - 1, dtype=bool, device=device), torch.zeros(N - 1, M - 1, dtype=bool, device=device)
            en_mask[torch.arange(N - 1), en_aligns] = True
            de_mask[de_aligns, torch.arange(M - 1)] = True
            en_ids, de_ids = torch.logical_and(en_mask, de_mask).nonzero(as_tuple=True)
            for dj, ej in zip(de_ids, en_ids):
                if de_words[i][dj] not in dictionary or attn[i][ej][dj] > attn_value.get(de_words[i][dj]):
                    dictionary[de_words[i][dj]] = en_words[i][ej]
                    attn_value[de_words[i][dj]] = attn[i][ej][dj]
    torch.save(dictionary, TextDataset.ADDITIONAL_DATA_FOLDER + 'de-en-dictionary.pt')

@torch.inference_mode()
def save_alignments(train_set: TextDataset, de_en_model, num_workers: int):
    batch_size = 64
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    device = next(de_en_model.parameters()).device
    de_en_model.eval()
    attns = []
    for de_indices, en_indices, de_lengths, en_lengths, de_texts, en_texts in tqdm(train_loader, desc='Collecting attentions'):
        de_indices, en_indices = de_indices.to(device), en_indices.to(device)
        de_words = [text.split() for text in de_texts]
        en_words = [text.split() for text in en_texts]
        B = de_indices.shape[0]
        attn = get_attention(de_en_model, de_indices, en_indices, de_lengths, en_lengths, False)
        
        for i in range(B):
            if len(de_words[i]) != de_lengths[i] - 2:
                raise AssertionError('Wrong de tokenization')
            if len(en_words[i]) != en_lengths[i] - 2:
                raise AssertionError('Wrong en tokenization')
            N, M = len(en_words[i]), len(de_words[i])
            de_aligns = attn[i][:N - 1, 1:M].argmax(dim=0)
            mask = torch.zeros(N - 1, M - 1, dtype=bool, device=device)
            mask[de_aligns, torch.arange(M - 1)] = True
            vals, aligns = torch.where(mask, attn[i][:N - 1, 1:M], float('-inf')).max(dim=1)
            aligns[vals == float('-inf')] = TextDataset.NO_ALIGN_INDEX
            attns.append(aligns)
            
    torch.save(attns, TextDataset.ADDITIONAL_DATA_FOLDER + 'training_alignments.pt')
    
@torch.inference_mode()
def get_masked_attention(model, de_indices, en_indices, de_lengths, en_lengths, swap_langs: bool = False, view_mode: bool = False):
    if not swap_langs:
        source, target, source_lengths, target_lengths = de_indices, en_indices, de_lengths, en_lengths
    else:
        source, target, source_lengths, target_lengths = en_indices, de_indices, en_lengths, de_lengths
    device = next(model.parameters()).device 
    source = source[:, :source_lengths.max() - 1].to(device)
    target = target[:, 1:target_lengths.max() - 1].to(device)
    source[source == model.dataset.de_eos_id] = model.dataset.de_pad_id
    target[target == model.dataset.de_eos_id] = model.dataset.de_pad_id
    preds = []
    attns = []
    for i in range(target_lengths.max().item() - 2):
        mask = torch.zeros_like(target, dtype=bool)
        mask[:, i] = True
        logits, attention = model(source, torch.where(mask, model.dataset.en_mask_id, target))
        attns.append(attention[:, i])
        if view_mode:
            preds.append(logits[0][i].argmax().item())

    if view_mode:
        return torch.stack(attns, dim=1)[0], preds
    else:
        return torch.stack(attns, dim=1)

@torch.inference_mode()
def view_alignments(train_set: TextDataset, de_en_model, num_preds: int = 3, masked: bool = False):
    device = next(de_en_model.parameters()).device
    de_en_model.eval()
    idx = torch.randint(len(train_set), size=(num_preds, ))
    for i in idx:
        de_ids, en_ids, de_lns, en_lns = train_set[i]
        de_text, en_text = train_set.de_texts[i], train_set.en_texts[i]
        if masked:
            de_en_attn, preds = get_masked_attention(de_en_model, de_ids.unsqueeze(0), en_ids.unsqueeze(0), torch.tensor(de_lns), torch.tensor(en_lns), False, True)
            print('Prediction:', train_set.ids2text(preds, 'en'))
        else:
            logits, de_en_attn = get_attention(de_en_model, de_ids.unsqueeze(0), en_ids.unsqueeze(0), torch.tensor(de_lns), torch.tensor(en_lns), False, True)
            print('Predictions:', train_set.ids2text(logits[0].argmax(dim=1).cpu(), 'en'))
            de_en_attn = de_en_attn[0]
        
        plt.matshow(de_en_attn.cpu())

        attn_values, aligns = de_en_attn.max(dim=1)
        de_words =  de_text.split()
        en_words = en_text.split()
        if len(de_words) != de_lns - 2:
            print('DE:', de_words)
            ids = train_set.text2ids(de_text, 'de')
            print([train_set.de_sp_model.IdToPiece(ind) for ind in ids])
            print('EN', de_text)
            raise AssertionError('Wrong de tokenization')
        if len(en_words) != en_lns - 2:
            raise AssertionError('Wrong en tokenization')
        print('DE:', de_text)
        print('EN:', en_text)
        #print(de_en_attn.shape, len(en_words), len(de_words))
        print('Alignments<-')
        for j in range(len(en_words)):
            if aligns[j] < de_lns - 1 and aligns[j] > 0:
                print(en_words[j], ':', de_words[aligns[j] - 1])
            elif aligns[j] == 0:
                print(en_words[j], ':', '<BOS>')
            else:
                raise AssertionError('Aligned too far')
        aligns2 = de_en_attn.argmax(dim=0)
        print('\n')
        print('Alignments->')
        for j in range(len(de_words)):
            print(de_words[j], ':', en_words[aligns2[j + 1]])


        vals = [[] for _ in range(len(en_words))]
        ids = [[] for _ in range(len(en_words))]
        for j in range(len(de_words)):
            ids[aligns2[j + 1]].append(j)
            vals[aligns2[j + 1]].append(de_en_attn[aligns2[j + 1]][j + 1].item())
        print('\n')
        print('Alignments<->')
        word_list = []
        for j in range(len(en_words)):
            if len(ids[j]) > 0:
                print(en_words[j], ':', de_words[ids[j][np.argmax(vals[j])]], end=' (')
                word_list.append(de_words[ids[j][np.argmax(vals[j])]])
            else:
                print(en_words[j], ':', '<NO>', end=' (')
                word_list.append('<NO>')
            
            if aligns[j] < de_lns - 1 and aligns[j] > 0:
                print(de_words[aligns[j] - 1], end='')
            elif aligns[j] == 0:
                print('<BOS>', end='')
            else:
                raise AssertionError('Aligned too far')
            print(')')
        print('\n')
        aligns3 = de_en_attn[:, 1:-1].argmax(dim=1)
        print('Alignments<-!')
        for j in range(len(en_words)):
            if aligns3[j] < de_lns - 2:
                print(en_words[j], ':', de_words[aligns3[j]], end='(')
            else:
                raise AssertionError('Aligned too far')
            print(word_list[j] + ')')
        print('\n')