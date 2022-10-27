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
            self.datamaps_df = pd.read_csv("/root/Langid/results/plots/datamaps-metrics-3.csv")
            # print(self.datamaps_df.head())
            self.easy_samples = self.datamaps_df[self.datamaps_df["confidence"]>0.6][self.datamaps_df["variability"]<0.3]
            self.hard_samples = self.datamaps_df[self.datamaps_df["confidence"]<0.3][self.datamaps_df["variability"]<0.2]
            self.ambiguous_samples = self.datamaps_df[~self.datamaps_df.isin(self.easy_samples)].dropna()
            self.ambiguous_samples = self.ambiguous_samples[~self.ambiguous_samples.isin(self.hard_samples)].dropna()
            
            # print(len(self.datamaps_df), len(self.easy_samples), len(self.hard_samples), len(self.ambiguous_samples))
            self.data = self.ambiguous_samples.values
        else:
            self.data = pd.read_csv(CSVPath).values

        # print(self.classes_set)
        self.is_train = is_train
        self.classes = {
            'ara-acm': torch.tensor([0.75,0.13139856,0.06841516,0.05018629,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]), 
            'ara-apc': torch.tensor([0.1090087,0.75,0.07234471,0.06864659,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]), 
            'ara-ary': torch.tensor([0.08757757,0.08694835,0.75,0.07547408,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]), 
            'ara-arz': torch.tensor([0.08027083,0.08976811,0.07996106,0.75,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]), 
            'eng-gbr': torch.tensor([0.,0.,0.,0.,0.75,0.25,0.,0.,0.,0.,0.,0.,0.,0.]), 
            'eng-usg': torch.tensor([0.,0.,0.,0.,0.25,0.75,0.,0.,0.,0.,0.,0.,0.,0.]), 
            'qsl-pol': torch.tensor([0.,0.,0.,0.,0.,0.,0.75,0.25,0.,0.,0.,0.,0.,0.]), 
            'qsl-rus': torch.tensor([0.,0.,0.,0.,0.,0.,0.25,0.75,0.,0.,0.,0.,0.,0.,]), 
            'por-brz': torch.tensor([0.,0.,0.,0.,0.,0.,0.,0.,0.75,0.08348278,0.08137903,0.08513819,0.,0.]), 
            'spa-car': torch.tensor([0.,0.,0.,0.,0.,0.,0.,0.,0.03007011,0.75,0.06198897,0.15794092,0.,0.]), 
            'spa-eur': torch.tensor([0.,0.,0.,0.,0.,0.,0.,0.,0.07484129,0.08263538,0.75,0.09252334,0.,0.]), 
            'spa-lac': torch.tensor([0.,0.,0.,0.,0.,0.,0.,0.,0.05206356,0.10113815,0.09679829,0.75,0.,0.,]), 
            'zho-cmn': torch.tensor([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.75,0.25]), 
            'zho-nan': torch.tensor([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.25,0.75])
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
        

# 25222 4681 773 19768
# 25222 6885 773 17564
# 24777 7662 814 16301
# 24757 7741 710 16306