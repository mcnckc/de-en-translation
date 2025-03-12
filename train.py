import os
import torch
from train_and_save_aligner import train_aligner_and_save
from employ_aligners import employ_aligner
from train_and_employ_gpt_ildus import train_gpt_ildus_and_predict
from adjustable_constants import NUM_WORKERS

if __name__ == '__main__': #deal with concurrency issues
    CHECKPOINTS_FOLDER = 'checkpoints' #Folder for models checkpoints, will be created automatically
    if not os.path.exists(CHECKPOINTS_FOLDER):
        os.mkdir(CHECKPOINTS_FOLDER)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print('Using device:', torch.cuda.get_device_name(device))
    else:
        print('А где куда????')
    
    train_aligner_and_save(device=device, folder=CHECKPOINTS_FOLDER, num_workers=NUM_WORKERS)
    employ_aligner(device=device, folder=CHECKPOINTS_FOLDER, num_workers=NUM_WORKERS)
    train_gpt_ildus_and_predict(device=device, checkpoints_folder=CHECKPOINTS_FOLDER, num_workers=NUM_WORKERS)


