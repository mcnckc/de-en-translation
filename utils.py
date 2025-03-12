import torch
import time
from dataset import TextDataset, TestDataset
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
from adjustable_constants import DATASET_FOLDER

def model_params(model):
    print('Trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    print('Total parameters:', sum(p.numel() for p in model.parameters()))

def view_predictions(model, val_set, num_preds=6, beam_size=0):
    device = next(model.parameters()).device
    idx = torch.randint(len(val_set), size=(num_preds, ))
    de_indices, de_lengths, en_texts = [], [], []
    for i in idx:
        de_ids, _, de_lns, _, en_text = val_set[i]
        de_indices.append(de_ids)
        de_lengths.append(de_lns)
        en_texts.append(en_text)
    de_indices = torch.vstack(de_indices).to(device)
    de_lengths = torch.tensor(de_lengths)
    model.eval()
    preds = model.inference(de_indices)
    ln = np.mean([len(s.split()) for s in preds])
    print('Average prediction length:', ln)
    if beam_size != 0:
        bs_preds = model.inference(de_indices, beam_size=beam_size)
        bln = np.mean([len(s.split()) for s in bs_preds])
        print(bs_preds[0].split())
        print('Average beam-search prediction length:', bln)
    refs = en_texts
    for i in range(num_preds):
        print(preds[i])
        if beam_size != 0:
            print(bs_preds[i])
        print(refs[i])
        print()

def predict(model, val_set: TextDataset, batch_size: int, num_workers: int, beam_size: int, lmbda: float):
    test_set = TestDataset(val_set)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    device = next(model.parameters()).device

    model.eval()
    preds = []
    for de_indices, de_lengths, de_texts in tqdm(test_loader, desc='Predicting'):
        de_indices = de_indices[:, :de_lengths.max()].to(device)
        preds += model.inference(de_indices, de_lengths, de_texts, beam_size, lmbda)
    
    with open('predictions.en', encoding='utf-8', mode='w') as file:
        file.writelines([pred + '\n' for pred in preds])

def save_model(model, name: str):
    torch.save({'model_state_dict': model.state_dict()}, name + '.pt')

def load_model(model, file: str):
    checkpoint = torch.load(file)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def average_models(models):
    n = len(models)
    ds = OrderedDict()
    for key in models[0].state_dict():
        ds[key] = torch.stack([model.state_dict()[key] for model in models]).mean(axis=0)
    models[0].load_state_dict(ds)
    return models[0]

def time_lengths(device):
    batch_size = 75
    train_set, _ = get_datasets(model_types=('word', 'word'), vocab_sizes=(20, 20), max_length=80, reverse_text=False)
    criterion = nn.CrossEntropyLoss(ignore_index=train_set.en_pad_id).to(device)
    lns = np.arange(10, 500, 2)
    times = []
    for ln in tqdm(lns):
        model = GPT(train_set, num_encoder_layers=2, num_decoder_layers=2, emb_size=64, nhead=4, dim_feedforward=32).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
        model.train()
        optimizer.zero_grad()
        de_indices = torch.full((batch_size, ln), 7, device=device)
        en_indices = torch.full((batch_size, ln), 7, device=device)
        torch.cuda.synchronize()
        st = time.time()
        logits = model(de_indices, en_indices[:, :-1])
        loss = criterion(logits.transpose(1, 2), en_indices[:, 1:])
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        end = time.time()
        times.append(end - st)

    return lns, times


class AttentionCollector:
    def __init__(self):
        self.attentions = []

    def __call__(self, module, module_in, module_out):
        self.attentions.append(module_out[1])

    def clear(self):
        self.outputs = []


def add_attention(model):

    real_forward = model.transformer.decoder.layers[-1].multihead_attn.forward

    def wrap(*args, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = True
        return real_forward(*args, **kwargs)

    model.transformer.decoder.layers[-1].multihead_attn.forward = wrap

    collector = AttentionCollector()
    hook = model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(collector)

    return collector