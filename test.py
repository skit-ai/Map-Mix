import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", UserWarning)

from config import LIDConfig
from Models.lightning import LightningModel
import pandas as pd
import torchaudio
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from argparse import ArgumentParser
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score


parser = ArgumentParser(add_help=True)
parser.add_argument('--train_path', type=str, default=LIDConfig.train_path)
parser.add_argument('--val_path', type=str, default=LIDConfig.val_path)
parser.add_argument('--test_path', type=str, default=LIDConfig.test_path)
parser.add_argument('--batch_size', type=int, default=LIDConfig.batch_size)
parser.add_argument('--epochs', type=int, default=LIDConfig.epochs)
parser.add_argument('--feature_dim', type=int, default=LIDConfig.feature_dim)
parser.add_argument('--lr', type=float, default=LIDConfig.lr)
parser.add_argument('--gpu', type=int, default=LIDConfig.gpu)
parser.add_argument('--n_workers', type=int, default=LIDConfig.n_workers)
parser.add_argument('--dev', type=str, default=False)
parser.add_argument('--model_checkpoint', type=str, default=LIDConfig.model_checkpoint)
parser.add_argument('--model_type', type=str, default=LIDConfig.model_type)
parser.add_argument('--upstream_model', type=str, default=LIDConfig.upstream_model)
parser.add_argument('--unfreeze_last_conv_layers', action='store_true')
parser.add_argument('--noise_dataset_path', type=str, default=None)

parser = pl.Trainer.add_argparse_args(parser)
hparams = parser.parse_args()

test_df = pd.read_csv(hparams.test_path)

results_df = pd.DataFrame(columns=['audiopath', 'duration', 'class', 'prediction', 'probability'])
model = LightningModel.load_from_checkpoint(hparams.model_checkpoint, HPARAMS=vars(hparams))
model.to('cuda')
model.eval()

num2labels = {0: 'ara-acm', 1: 'ara-apc', 2: 'ara-ary', 3: 'ara-arz', 4: 'eng-gbr', 5: 'eng-usg', 6: 'qsl-pol', 7: 'qsl-rus', 8: 'por-brz', 9: 'spa-car', 10: 'spa-eur', 11: 'spa-lac', 12: 'zho-cmn', 13: 'zho-nan'} 
def convert_to_labels(num):
            return num2labels[num]

trues=[]
preds = []

for index, row in tqdm(test_df.iterrows()):
    file_path = row["audiopath"]
    true_label = row["class"]
    duration = row["seconds"]

    wav, _ = torchaudio.load(file_path)
    wav = wav.view(-1)

    if wav.shape[-1] > 16000*6:
        wav_tensor = wav.unfold(0, 16000*6, 16000*3)
    else:
        wav_tensor = wav.view(1, -1)

    wav_tensor = wav_tensor.to("cuda")
    x_lens = wav_tensor.shape[0]*[wav_tensor.shape[-1]]
    y_hat_l = model(wav_tensor, x_lens)
    probs = F.softmax(y_hat_l, dim=1).detach().cpu().mean(0).view(1, 14)
    y_hat_l = probs.argmax(dim=1).detach().cpu().numpy().astype(int)
    probs = probs.numpy().astype(float).tolist()

    predictions = list(map(convert_to_labels, y_hat_l))
    ground_truths = [true_label]
    trues.append(ground_truths[0])
    preds.append(predictions[0])

    rows = {'audiopath': file_path, 'duration': duration, 'class': ground_truths, 'prediction': predictions, 'probability': probs}
    results_df = results_df.append(pd.DataFrame(rows))

results_df.to_csv(LIDConfig.results_path, index=False)
print(accuracy_score(trues, preds), f1_score(trues, preds, average='weighted'))