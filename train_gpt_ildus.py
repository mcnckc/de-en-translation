import torch
import numpy as np
from typing import List, Optional, Any
from torch import nn
from torch.utils.data import DataLoader
from sacrebleu.metrics import BLEU
from tqdm import tqdm

#import wandb
#import seaborn as sns
#import matplotlib.pyplot as plt
#from IPython.display import clear_output

"""
sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 15})

def plot_losses(train_losses: List[float], val_losses: List[float], val_bleus: List[float], val_bs_bleus, ref_lens, pred_lens, bs_pred_lens, probs, bs_probs, t_probs, train_align_losses):
    clear_output()
    fig, axs = plt.subplots(3, 2, figsize=(14, 12))
    axs[0][0].plot(range(1, len(train_losses) + 1), train_losses, label='train')
    axs[0][0].plot(range(1, len(val_losses) + 1), val_losses, label='val')
    axs[0][0].set_ylabel('loss')

    axs[0][1].plot(range(1, len(val_bleus) + 1), val_bleus, label='val')
    axs[0][1].plot(range(1, len(val_bleus) + 1), val_bs_bleus, label='val beam search', color=u'#ff7f0e')
    axs[0][1].set_ylabel('BLEU Score')

    axs[1][0].plot(range(1, len(val_bleus) + 1), ref_lens, label='reference')
    axs[1][0].plot(range(1, len(val_bleus) + 1), pred_lens, label='predictions')
    axs[1][0].plot(range(1, len(val_bleus) + 1), bs_pred_lens, label='beam search predictions')
    axs[1][0].set_ylabel('Average word length')

    axs[1][1].plot(range(1, len(val_bleus) + 1), t_probs, label='reference')
    axs[1][1].plot(range(1, len(val_bleus) + 1), probs, label='predictions')
    axs[1][1].plot(range(1, len(val_bleus) + 1), bs_probs, label='beam search predictions')
    axs[1][1].set_ylabel('Log of estimated probabilities')

    axs[2][0].plot(range(1, len(val_bleus) + 1), train_align_losses, label='train')
    axs[2][0].set_ylabel('Alignment loss')
    
    for axi in axs:
        for ax in axi:
            ax.set_xlabel('epoch')
            ax.legend()

    plt.show()
"""

def training_epoch(model: nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module, align_criterion: nn.Module,
                   loader: DataLoader, tqdm_desc: str, scheduler: torch.optim.lr_scheduler.LambdaLR, align_w: float):
    device = next(model.parameters()).device
    train_loss = 0.0
    train_align_loss = 0.0
    avg_max = 0.0
    for de_indices, en_indices, de_lengths, en_lengths, aligns in tqdm(loader, desc=tqdm_desc, leave=False):
        model.train()
        optimizer.zero_grad()
        de_indices = de_indices[:, :de_lengths.max()].to(device)
        en_indices = en_indices[:, :en_lengths.max()].to(device)
        aligns = aligns[:, :en_lengths.max()].to(device)
        logits, attns = model(de_indices, en_indices[:, :-1])
        loss = criterion(logits.transpose(1, 2), en_indices[:, 1:])
        align_loss = align_criterion(attns.transpose(1, 2), aligns[:, 1:])
        loss += align_loss * align_w
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * de_indices.shape[0]
        train_align_loss += align_loss.item() * de_indices.shape[0]
        if scheduler is not None:
            scheduler.step()
    train_loss /= len(loader.dataset)
    train_align_loss /= len(loader.dataset)
    return train_loss, train_align_loss


@torch.no_grad()
def validation_epoch(model: nn.Module, criterion: nn.Module,
                     loader: DataLoader, tqdm_desc: str, beam_size: int, 
                     lmbda: float, align_w: float):
    device = next(model.parameters()).device
    val_loss = 0.0

    model.eval()
    bleu = BLEU()
    preds = []
    bs_preds = []
    refs = []
    t_prob = 0.0
    prob = 0.0
    bs_prob = 0.0
    unks = 0
    for de_indices, en_indices, de_lengths, en_lengths, de_texts, en_texts in tqdm(loader, desc=tqdm_desc, leave=False):
        de_indices = de_indices[:, :de_lengths.max()].to(device)
        en_indices = en_indices[:, :en_lengths.max()].to(device)
        logits, attns = model(de_indices, en_indices[:, :-1])
        loss = criterion(logits.transpose(1, 2), en_indices[:, 1:])
        val_loss += loss.item() * de_indices.shape[0]

        new_preds, new_indices = model.inference(de_indices, de_lengths, de_texts, return_indices=True)
        preds += new_preds
        prob += model.estimate_likelyhood(de_indices, new_indices)
        unks += (new_indices == model.dataset.en_unk_id).sum().item()

        new_preds, new_indices = model.inference(de_indices, de_lengths, de_texts, beam_size=beam_size, lmbda=lmbda, return_indices=True)
        bs_preds += new_preds
        bs_prob += model.estimate_likelyhood(de_indices, new_indices)

        refs += en_texts
        t_prob += model.estimate_likelyhood(de_indices, en_indices)

    val_loss /= len(loader.dataset)
    prob /= len(loader.dataset)
    bs_prob /= len(loader.dataset)
    t_prob /= len(loader.dataset)
    unks /= len(loader.dataset)
    ref_len = np.mean([len(s.split()) for s in refs])
    pred_len = np.mean([len(s.split()) for s in preds])
    bs_pred_len = np.mean([len(s.split()) for s in bs_preds])
    return val_loss, bleu.corpus_score(preds, [refs]).score, bleu.corpus_score(bs_preds, [refs]).score, ref_len, pred_len, bs_pred_len, prob, bs_prob, t_prob, unks


def train(model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: Optional[Any],
          train_loader: DataLoader, val_loader: DataLoader, num_epochs: int, save_epoch: int = 1000, 
          beam_size: int = 0, lmbda: float = 0, align_w: float = 0, smoothing: float = 0, bs_epoch: int = 1,
          checkpoints_folder: str = 'checkpoints'):
    train_losses, train_align_losses, val_losses, val_bleus = [], [], [], []
    val_bs_bleus, ref_lens, pred_lens, bs_pred_lens, probs, bs_probs, t_probs = [], [], [], [], [], [], []
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss(ignore_index=train_loader.dataset.en_pad_id, label_smoothing=smoothing).to(device)
    align_criterion = nn.CrossEntropyLoss(ignore_index=train_loader.dataset.NO_ALIGN_INDEX).to(device)

    for epoch in range(1, num_epochs + 1):
        train_loss, train_align_loss = training_epoch(
            model, optimizer, criterion, align_criterion, train_loader,
            tqdm_desc=f'Training {epoch}/{num_epochs}', scheduler=scheduler, align_w=align_w
        )
        
        val_loss, val_bleu, val_bs_bleu, ref_len, pred_len, bs_pred_len, prob, bs_prob, t_prob, unks = validation_epoch(
            model, criterion, val_loader,
            tqdm_desc=f'Validating {epoch}/{num_epochs}', beam_size=beam_size if epoch >= bs_epoch else 0, lmbda=lmbda, align_w=align_w
        )
        
        #if scheduler is not None:
            #scheduler.step(val_loss)

        train_losses += [train_loss]
        train_align_losses += [train_align_loss]
        val_losses += [val_loss]
        val_bleus += [val_bleu]

        val_bs_bleus += [val_bs_bleu]
        ref_lens += [ref_len]
        pred_lens += [pred_len]
        bs_pred_lens += [bs_pred_len]
        probs += [prob]
        bs_probs += [bs_prob]
        t_probs += [t_prob]

        """
        wandb.log({'Train loss':train_loss, 'Test loss': val_loss, 'Test BLEU':val_bleu, 'Learning rate':optimizer.param_groups[0]['lr'], 
                   'Test beam-search BLEU':val_bs_bleu, 'Average ref length':ref_len, 'Average prediction length': pred_len, 
                   'Average Beam Search predction length': bs_pred_len, 'Average unk per sentence':unks})
        plot_losses(train_losses, val_losses, val_bleus, val_bs_bleus, ref_lens, pred_lens, bs_pred_lens, probs, bs_probs, t_probs, train_align_losses)
        """

        if epoch >= save_epoch:
            torch.save({
            'model_state_dict': model.state_dict()
            }, checkpoints_folder + '/gpt-ildus-' + str(epoch) + '.pt')
    
    return train_losses, val_losses, val_bleus


def get_lr_lambda(d_model: int, warmup: int):
    d_model = d_model ** (-0.5)
    warmup = warmup ** (-1.5)
    return lambda step: d_model * min((step + 1) ** (-0.5), (step + 1) * warmup)

def get_lr_lambda2(warmup: int):
    return lambda step: 1 if step <= warmup else (step - warmup) ** (-0.5)
