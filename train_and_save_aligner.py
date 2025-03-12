import torch
from dataset import get_datasets
from aligner import Aligner
from utils import model_params, load_model
from train_aligner import train, freeze_for_ft
from torch.utils.data import DataLoader

def train_aligner_and_save(device, folder: str, num_workers: int, num_epochs: int = 400, from_checkpoint: bool = False, ft: bool = True):
    train_set, valid_set = get_datasets(sub_sample=1, model_types=('word', 'word'), vocab_sizes=(20000, 10000), full_vocabs=(True, False), max_length=82, reverse_text=False)
    model = Aligner(train_set, num_encoder_layers=3, num_decoder_layers=3, emb_size=192, nhead=6, dim_feedforward=384, dropout=0.055).to(device)
    if from_checkpoint :
        model = load_model(model, folder + '/aligner_model.pt')
    print('Training Aligner with:')
    model_params(model)
    print('\n')
    batch_size = 400
    epochs = num_epochs #probably 250-300 can be enough, but score may drop
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    train_loss, val_loss, train_accs, val_accs = train(model, optimizer, None, train_loader=train_loader, val_loader=val_loader, num_epochs=epochs, mask_rate=0.13)
    
    if not ft:
        torch.save({
            'model_state_dict': model.state_dict()
        }, folder + '/aligner_model.pt' )
        return
    
    freeze_for_ft(model)
    print('Fine tuning aligner with:')
    model_params(model)
    print('\n')
    epochs = 12
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train_loss, val_loss, train_accs, val_accs = train(model, optimizer, None, train_loader=train_loader, val_loader=val_loader, num_epochs=epochs, fine_tuning=True)
    torch.save({
        'model_state_dict': model.state_dict()
    }, folder + '/aligner_model.pt' )


