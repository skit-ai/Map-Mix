import pandas as pd
import random

import torch
from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn_utils

import torchaudio
import wavencoder
import librosa
import numpy as np

def collate_fn_mixup(batch):
    (seq, mixup_seq, label, mixup_label, wav_duration, filename) = zip(*batch)
    
    seql = [x.reshape(-1,) for x in seq]
    mixup_seql = [x.reshape(-1,) for x in mixup_seq]

    seq_length = [x.shape[0] for x in seql]
    mixup_seq_length = [x.shape[0] for x in mixup_seql]
    
    data = rnn_utils.pad_sequence(seql, batch_first=True, padding_value=0)
    mixup_data = rnn_utils.pad_sequence(mixup_seql, batch_first=True, padding_value=0)
    return data, mixup_data, label, mixup_label, seq_length, mixup_seq_length, filename

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
    cluster = "across"
    ):
        
        

        self.CSVPath = CSVPath

        if is_train:
            self.datamaps_df = pd.read_csv("/root/Langid/results/plots/datamaps-metrics-2.csv")
            # print(self.datamaps_df.head())
            self.easy_samples = self.datamaps_df[self.datamaps_df["confidence"]>0.65][self.datamaps_df["variability"]<0.3]
            self.hard_samples = self.datamaps_df[self.datamaps_df["confidence"]<0.3][self.datamaps_df["variability"]<0.2]
            self.ambiguous_samples = self.datamaps_df[~self.datamaps_df.isin(self.easy_samples)].dropna()
            self.ambiguous_samples = self.ambiguous_samples[~self.ambiguous_samples.isin(self.hard_samples)].dropna()
            
            self.data = self.ambiguous_samples.values
        else:
            self.data = pd.read_csv(CSVPath).values

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
        self.train_transform = wavencoder.transforms.PadCrop(pad_crop_length=16000*8, pad_position='random', crop_position='random')
        self.test_transform = wavencoder.transforms.PadCrop(pad_crop_length=16000*20, pad_position='left', crop_position='center')
        self.cluster = cluster

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

        # Mixup
        mixup_wav = torch.zeros(1)
        mixup_language = torch.zeros(14)

        ######## Applying Mixup #########
        probability = 1.0
        if self.is_train:
            if random.random() <= probability:
                l = len(self.easy_samples)
                i = random.randint(0, l-1)
                mixup_sample = self.easy_samples.iloc[i]
                mixup_file = mixup_sample['audiopath']
                mix_class =  mixup_sample['class']
                mixup_language = self.classes[mix_class]

                mixup_wav, _ = torchaudio.load(mixup_file)
                mixup_wav = self.train_transform(mixup_wav)
        ######## Done applying Mixup #########
            wav = self.train_transform(wav)
            if random.random()>0.5:
            # Time Mask
                l = wav.shape[-1]
                window = random.choice([16000, 2*16000, 3*16000, 4*16000])
                t = int(np.random.uniform(low=0, high=l-window))
                wav[:,  t: t+window] = 0
            return wav, mixup_wav, language, mixup_language, torch.FloatTensor([wav_duration]), file
        else:
            wav = self.test_transform(wav)
            return wav, torch.FloatTensor([wav_duration]), language


if __name__ == "__main__":
    dataset = LIDDataset(
        CSVPath = "/root/LRE2017Dataset/LRE2017/lre17-segmented-train-set-5-hour-each-set-1.csv",
        hparams = None,
        is_train=True,)
    _ = dataset[0]
        
