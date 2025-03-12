import torch
from dataset import get_datasets
from gpt_ildus import GPT_ILDUS
from utils import model_params, average_models, predict
from torch.utils.data import DataLoader
from train_gpt_ildus import train, get_lr_lambda

def train_gpt_ildus_and_predict(device, checkpoints_folder: str, num_workers: int):
    train_set, valid_set = get_datasets(sub_sample=1, model_types=('word', 'word'), vocab_sizes=(30000, 20000), 
                                        full_vocabs=(False, False), max_length=82, mode='aligns',
                                        enable_dictionary=True, force_training=True)
    model = GPT_ILDUS(train_set, num_encoder_layers=4, num_decoder_layers=4, emb_size=360, nhead=8, dim_feedforward=512, dropout=0.12).to(device)
    print('Training translator with:')
    model_params(model)
    batch_size = 220
    epochs = 60
    save_epoch = 41
    optimizer = torch.optim.Adam(model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_lambda(360, 4000))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    train_loss, val_loss, train_accs = train(model, optimizer, scheduler, train_loader=train_loader, val_loader=val_loader, 
                                             num_epochs=epochs, beam_size=0, lmbda=1, align_w=0.05, smoothing=0.05, 
                                             save_epoch=save_epoch, checkpoints_folder=checkpoints_folder)
    models = []
    for i in range(save_epoch, epochs + 1):
        models.append(GPT_ILDUS(train_set, num_encoder_layers=4, num_decoder_layers=4, emb_size=360, nhead=8, dim_feedforward=512, dropout=0.12).to(device))
        checkpoint = torch.load(checkpoints_folder + '/gpt-ildus-' + str(i) + '.pt')
        models[-1].load_state_dict(checkpoint['model_state_dict'])

    final_model = average_models(models)
    predict(final_model, valid_set, batch_size, num_workers, beam_size=40, lmbda=1)
    
   
    
    

