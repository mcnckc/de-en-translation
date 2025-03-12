from utils import predict, load_model
import argparse
from gpt_ildus import GPT_ILDUS
import torch
from dataset import get_datasets
from adjustable_constants import NUM_WORKERS

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser("predictor")
    parser.add_argument("--checkpoint", dest="checkpoint", help="path to model checkpoint", type=str)
    parser.add_argument("--beam_size", dest="beam_size", nargs='?', const=40, help="Beam size in beam search", type=int)
    args = parser.parse_args()
    batch_size = 64
    train_set, valid_set = get_datasets(sub_sample=1, model_types=('word', 'word'), vocab_sizes=(30000, 20000), 
                                        full_vocabs=(False, False), max_length=82, mode='aligns',
                                        enable_dictionary=True, force_training=True)
    model = GPT_ILDUS(train_set, num_encoder_layers=4, num_decoder_layers=4, emb_size=360, nhead=8, dim_feedforward=512, dropout=0.12).to(device)
    load_model(model, args.checkpoint)
    predict(model, valid_set, batch_size, NUM_WORKERS, beam_size=args.beam_size, lmbda=1)
