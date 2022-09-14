import pandas as pd
import random

import torch
from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn_utils

import torchaudio
import wavencoder
import librosa

def collate_fn(batch):
    (seq, wav_duration, label, filenames) = zip(*batch)
    
    seql = [x.reshape(-1,) for x in seq]

    seq_length = [x.shape[0] for x in seql]
    
    data = rnn_utils.pad_sequence(seql, batch_first=True, padding_value=0)
    return data, label, seq_length, filenames


class LIDDataset(Dataset):
    def __init__(self,
    CSVPath,
    hparams,
    is_train=True,
    ):
        self.CSVPath = CSVPath
        self.data = pd.read_csv(CSVPath).values
        if is_train:
            self.datacsv = pd.read_csv(CSVPath)
            self.datacsv['language'] = self.datacsv['class'].astype(str).str[:3]
            self.datacsv['dialect'] = self.datacsv['class'].astype(str).str[4:]
            self.classes_set = set(self.datacsv["class"].values)
            self.lang_set = set(self.datacsv["language"].values)
            self.dia_set = set(self.datacsv["dialect"].values)

        # print(self.classes_set)
        self.is_train = is_train
        self.classes = {
            'ara-acm': torch.eye(14)[0], 
            'ara-apc': torch.eye(14)[1], 
            'ara-ary': torch.eye(14)[2], 
            'ara-arz': torch.eye(14)[3], 
            'eng-gbr': torch.eye(14)[4], 
            'eng-usg': torch.eye(14)[5], 
            'qsl-pol': torch.eye(14)[6], 
            'qsl-rus': torch.eye(14)[7], 
            'por-brz': torch.eye(14)[8], 
            'spa-car': torch.eye(14)[9], 
            'spa-eur': torch.eye(14)[10], 
            'spa-lac': torch.eye(14)[11], 
            'zho-cmn': torch.eye(14)[12], 
            'zho-nan': torch.eye(14)[13]
            }
        self.lang2cluster = {0:1, 1:1, 2:1, 3:1, 4:2, 5:2, 6:3, 7:3, 8:4, 9:4, 10:4, 11:4, 12:5, 13:5}
        self.train_transform = wavencoder.transforms.PadCrop(pad_crop_length=8*16000, pad_position='random', crop_position='random')
        self.test_transform = wavencoder.transforms.PadCrop(pad_crop_length=20*16000, pad_position='left', crop_position='center')

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        file = self.data[idx][0]
        
        language = self.classes[self.data[idx][1]]

        wav, _ = torchaudio.load(file)
        
        if(self.data.shape[1] == 3):
            wav_duration = self.data[idx][2]
        else:
            wav_duration = -1

        if(self.is_train):
            wav = self.train_transform(wav)
        else:
            wav = self.test_transform(wav)
        return wav, torch.FloatTensor([wav_duration]), language, file


if __name__ == "__main__":
    dataset = LIDDataset(
        CSVPath = "/root/LRE2017Dataset/LRE2017/lre17-segmented-train-set-5-hour-each-set-1.csv",
        hparams = None,
        is_train=True,)
    _ = dataset[0]
        
