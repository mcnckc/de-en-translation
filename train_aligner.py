import torch
import numpy as np
from typing import List, Optional, Any
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
    
#import wandb
#import seaborn as sns
#import matplotlib.pyplot as plt
#from IPython.display import clear_output

"""
sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 15})


def plot_losses(train_losses: List[float], val_losses: List[float], train_accs: List[float], val_accs: List[float]):
    clear_output()
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    axs[0].plot(range(1, len(train_losses) + 1), train_losses, label='train')
    axs[0].plot(range(1, len(val_losses) + 1), val_losses, label='val')
    axs[0].set_ylabel('loss')

    axs[1].plot(range(1, len(train_accs) + 1), train_accs, label='train')
    axs[1].plot(range(1, len(val_accs) + 1), val_accs, label='val')
    axs[1].set_ylabel('Mask accuracy')

    for ax in axs:
        ax.set_xlabel('epoch')
        ax.legend()

    plt.show()
"""

def mask_batch(indices: torch.Tensor, lenghts: torch.Tensor, mask_rate: float) -> torch.Tensor:
    true_lenghts = lenghts - 2
    mask = torch.zeros_like(indices, dtype=bool, device=indices.device)
    for i in range(indices.shape[0]):
        k = max(np.rint(true_lenghts[i].item() * mask_rate).astype(int), 1)
        mask[i, torch.randperm(true_lenghts[i].item(), device=indices.device)[:k]] = True
    return mask

def training_epoch(model: nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module,
                   loader: DataLoader, tqdm_desc: str, scheduler: torch.optim.lr_scheduler.LambdaLR, 
                   mask_rate: float, swap_langs: bool, fine_tuning: bool):
    device = next(model.parameters()).device
    train_loss = 0.0
    guessed = 0.0
    total_masks = 0
    for de_indices, en_indices, de_lengths, en_lengths in tqdm(loader, desc=tqdm_desc, leave=False):
        model.train()
        optimizer.zero_grad()
        if not swap_langs:
            source, target, source_lengths, target_lengths = de_indices, en_indices, de_lengths, en_lengths
        else:
            source, target, source_lengths, target_lengths = en_indices, de_indices, en_lengths, de_lengths

        source = source[:, :source_lengths.max() - 1].to(device)
        target = target[:, 1:target_lengths.max() - 1].to(device)
        source[source == model.dataset.de_eos_id] = model.dataset.de_pad_id
        target[target == model.dataset.en_eos_id] = model.dataset.en_pad_id
        if not fine_tuning:
            mask = mask_batch(target, target_lengths, mask_rate)
            logits, _ = model(source, torch.where(mask, model.dataset.en_mask_id, target))
            loss = criterion(logits.transpose(1, 2), torch.where(mask, target, model.dataset.en_pad_id))
        else:
            logits, _ = model(source, target)
            loss = criterion(logits.transpose(1, 2), target)
        loss.backward()
        optimizer.step()
        preds = logits.argmax(dim=2)
        if not fine_tuning:
            guessed += torch.logical_and(preds == target, mask).sum()
            total_masks += mask.sum()
        else:
            mask = torch.zeros_like(target, dtype=bool)
            for i in range(target.shape[0]):
                mask[i, :target_lengths[i] - 2] = True
            guessed += torch.logical_and(preds == target, mask).sum()
            total_masks += (target_lengths - 2).sum()

        train_loss += loss.item() * source.shape[0]
        if scheduler is not None:
            scheduler.step()
    train_loss /= len(loader.dataset)
    guessed /= total_masks
    return train_loss, guessed.item()


@torch.no_grad()
def validation_epoch(model: nn.Module, criterion: nn.Module,
                     loader: DataLoader, tqdm_desc: str, mask_rate: float, 
                     swap_langs: bool, fine_tuning: bool):
    device = next(model.parameters()).device
    val_loss = 0.0
    guessed = 0.0
    total_masks = 0
    model.eval()
    unks = 0.0
    tunks = 0.0
    for de_indices, en_indices, de_lengths, en_lengths, en_texts in tqdm(loader, desc=tqdm_desc, leave=False):
        if not swap_langs:
            source, target, source_lengths, target_lengths = de_indices, en_indices, de_lengths, en_lengths
        else:
            source, target, source_lengths, target_lengths = en_indices, de_indices, en_lengths, de_lengths

        source = source[:, :source_lengths.max() - 1].to(device)
        target = target[:, 1:target_lengths.max() - 1].to(device)
        source[source == model.dataset.de_eos_id] = model.dataset.de_pad_id
        target[target == model.dataset.en_eos_id] = model.dataset.en_pad_id
        if not fine_tuning:
            mask = mask_batch(target, target_lengths, mask_rate)
            logits, _ = model(source, torch.where(mask, model.dataset.en_mask_id, target))
            loss = criterion(logits.transpose(1, 2), torch.where(mask, target, model.dataset.en_pad_id))
        else:
            logits, _ = model(source, target)
            loss = criterion(logits.transpose(1, 2), target)

        preds = logits.argmax(dim=2)
        if not fine_tuning:
            guessed += torch.logical_and(preds == target, mask).sum()
            total_masks += mask.sum()
            unks += torch.logical_and(preds == model.dataset.en_unk_id, mask).sum().item()
        else:
            mask = torch.zeros_like(target, dtype=bool)
            for i in range(target.shape[0]):
                mask[i, :target_lengths[i] - 2] = True
            guessed += torch.logical_and(preds == target, mask).sum()
            total_masks += (target_lengths - 2).sum()
            unks += torch.logical_and(preds == model.dataset.en_unk_id, mask).sum().item()
        tunks += (target == model.dataset.en_unk_id).sum().item()
        val_loss += loss.item() * source.shape[0]
        
    val_loss /= len(loader.dataset)
    guessed /= total_masks
    unks /= total_masks.item()
    tunks /= len(loader.dataset)
    return val_loss, guessed.item(), unks, tunks


def train(model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: Optional[Any],
          train_loader: DataLoader, val_loader: DataLoader, num_epochs: int, save_epoch: int = 1000, 
          mask_rate: float = 0.1, swap_langs: bool = False, fine_tuning: bool = False):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss(ignore_index=train_loader.dataset.en_pad_id).to(device)

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = training_epoch(
            model, optimizer, criterion, train_loader,
            tqdm_desc=f'Training {epoch}/{num_epochs}', scheduler=scheduler, 
            mask_rate=mask_rate, swap_langs=swap_langs, fine_tuning=fine_tuning
        )
        
        val_loss, val_acc, unks, tunks = validation_epoch(
            model, criterion, val_loader,
            tqdm_desc=f'Validating {epoch}/{num_epochs}', 
            mask_rate=mask_rate, swap_langs=swap_langs,
            fine_tuning=fine_tuning
        )
        
        #if scheduler is not None:
            #scheduler.step(val_loss)

        train_losses += [train_loss]
        val_losses += [val_loss]
        train_accs += [train_acc]
        val_accs += [val_acc]
        #wandb.log({'Train loss':train_loss, 'Test loss': val_loss, 'Test BLEU':val_bleu, 'Learning rate':optimizer.param_groups[0]['lr'], 
                   #'Test beam-search BLEU':val_bs_bleu, 'Average ref length':ref_len, 'Average prediction length': pred_len, 
                   #'Average Beam Search predction length': bs_pred_len, 'Average unk per sentence':unks})
        #plot_losses(train_losses, val_losses, train_accs, val_accs)
    
    return train_losses, val_losses, train_accs, val_accs

def freeze_for_ft(model):
    for p in model.parameters():
        p.requires_grad = False
    model.transformer.decoder.layers[-1].multihead_attn.q_proj_weight.requires_grad = True
    model.transformer.decoder.layers[-1].multihead_attn.k_proj_weight.requires_grad = True

def get_lr_lambda(d_model: int, warmup: int):
    d_model = d_model ** (-0.5)
    warmup = warmup ** (-1.5)
    return lambda step: d_model * min((step + 1) ** (-0.5), (step + 1) * warmup)

def get_lr_lambda2(warmup: int):
    return lambda step: 1 if step <= warmup else (step - warmup) ** (-0.5)
