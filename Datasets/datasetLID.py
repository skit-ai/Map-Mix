import pandas as pd

import torch
from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn_utils

import torchaudio

import wavencoder
import librosa
import random
import numpy as np

def collate_fn(batch):
    (seq, wav_duration, label) = zip(*batch)
    seql = [x.reshape(-1,) for x in seq]
    seq_length = [x.shape[0] for x in seql]
    data = rnn_utils.pad_sequence(seql, batch_first=True, padding_value=0)
    return data, seq_length, label

class LIDDataset(Dataset):
    def __init__(self,
    CSVPath,
    hparams,
    is_train=True,
    ):
        self.CSVPath = CSVPath
        self.data = pd.read_csv(CSVPath).values
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
        # self.upsample = torchaudio.transforms.Resample(orig_freq=22050, new_freq=16000)
        self.train_transform = wavencoder.transforms.PadCrop(pad_crop_length=16000*8, pad_position='random', crop_position='random')
        self.test_transform = wavencoder.transforms.PadCrop(pad_crop_length=16000*20, pad_position='left', crop_position='center')

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        file = self.data[idx][0]
        if file.endswith(".wav.wav"):
            file = file[:-4]
        language = self.classes[self.data[idx][1]]

        # wav, _ = librosa.load(file)
        wav, _ = torchaudio.load(file)
        # wav = torch.from_numpy(wav)
        
        if(self.data.shape[1] == 3):
            wav_duration = self.data[idx][2]
        else:
            wav_duration = -1

        # upsample 8k -> 16k
        # wav = self.upsample(wav).unsqueeze(dim=0) 
        # wav = wav.unsqueeze(dim=0)

        if(self.is_train):
            wav = self.train_transform(wav)

            if random.random()>0.5:
            # Time Mask
                l = wav.shape[-1]
                window = random.choice([16000, 2*16000, 3*16000, 4*16000])
                t = int(np.random.uniform(low=0, high=l-window))
                wav[:,  t: t+window] = 0

        else:
            wav = self.test_transform(wav)
        return wav, torch.FloatTensor([wav_duration]), language
