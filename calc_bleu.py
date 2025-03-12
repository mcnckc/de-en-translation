import sys
from sacrebleu.metrics import BLEU

if __name__ == '__main__':
    true_file, pred_file = sys.argv[1], sys.argv[2]
    preds = []
    refs = []
    with open(true_file, 'r') as f:
        refs = [line.rstrip() for line in f]
    with open(pred_file, 'r') as f:
        preds = [line.rstrip() for line in f]
    bleu = BLEU()
    print("BLUE SCORE:", bleu.corpus_score(preds, [refs]).score)