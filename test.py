from argparse import ArgumentParser

from scipy.special import softmax
import pandas as pd
import pytorch_lightning as pl

import torch
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import torch.utils.data as data

from tqdm import tqdm

from config import LIDConfig
from Datasets.datasetLID import LIDDataset
from Models.lightning import LightningModel

# def collate_fn(batch):
#     (seq, wav_durations, label) = zip(*batch)
#     seql = [x.reshape(-1,) for x in seq]
#     seq_length = [x.shape[0] for x in seql]
#     data = rnn_utils.pad_sequence(seql, batch_first=True, padding_value=0)
#     return data, wav_durations, label

def collate_fn(batch):
    (seq, wav_duration, label) = zip(*batch)
    seql = [x.reshape(-1,) for x in seq]
    seq_length = [x.shape[0] for x in seql]
    data = rnn_utils.pad_sequence(seql, batch_first=True, padding_value=0)
    return data, seq_length, label

if __name__ == "__main__":

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
    parser.add_argument('--mixup_type', type=str, default=LIDConfig.mixup_type)
    parser.add_argument('--cluster', type=str, default=LIDConfig.cluster)
    
    parser = pl.Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()
    print(f'Testing Model on LID Dataset\n#Cores = {hparams.n_workers}\t#GPU = {hparams.gpu}')

    # Testing Dataset
    test_set = LIDDataset(
        CSVPath = hparams.test_path,
        hparams = hparams,
        is_train=False
    )
    
    ## Testing Dataloader
    testloader = data.DataLoader(
        test_set, 
        batch_size=1, 
        shuffle=False, 
        num_workers=hparams.n_workers,
        collate_fn = collate_fn,
    )

    num2labels = {0: 'ara-acm', 1: 'ara-apc', 2: 'ara-ary', 3: 'ara-arz', 4: 'eng-gbr', 5: 'eng-usg', 6: 'qsl-pol', 7: 'qsl-rus', 8: 'por-brz', 9: 'spa-car', 10: 'spa-eur', 11: 'spa-lac', 12: 'zho-cmn', 13: 'zho-nan'} 

    #Testing the Model
    if hparams.model_checkpoint:
        results_df = pd.DataFrame(columns=['audiopath', 'duration', 'class', 'prediction', 'probability'])

        model = LightningModel.load_from_checkpoint(hparams.model_checkpoint, HPARAMS=vars(hparams))
        model.to('cuda')
        model.eval()

        def convert_to_labels(num):
            return num2labels[num]

    
        for batch in tqdm(testloader):
            apply_mixup = False
            x, x_len, y_l  = batch
            x = x.to('cuda')

            y_l = torch.stack(y_l)
            y_hat_l = model.simple_forward(x, x_len)

            probs = F.softmax(y_hat_l, dim=1).detach().cpu()
            probs = probs.numpy().astype(float).tolist()

            y_l = y_l.argmax(dim=1).detach().cpu().numpy().astype(int)
            y_hat_l = y_hat_l.argmax(dim=1).detach().cpu().numpy().astype(int)

            ground_truths = list(map(convert_to_labels, y_l))
            predictions = list(map(convert_to_labels, y_hat_l))

            rows = {'class': ground_truths, 'prediction': predictions, 'probability': probs}

            results_df = results_df.append(pd.DataFrame(rows))

        results_df.to_csv(LIDConfig.results_path, index=False)
    else:
        print('Model checkpoint not found for Testing !!!')