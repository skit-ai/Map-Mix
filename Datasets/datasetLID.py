import pandas as pd
import random

import torch
from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn_utils

import torchaudio
import wavencoder
import librosa

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
        self.upsample = torchaudio.transforms.Resample(orig_freq=8000, new_freq=16000)
        self.train_transform = wavencoder.transforms.PadCrop(pad_crop_length=480000, pad_position='random', crop_position='random')
        self.test_transform = wavencoder.transforms.PadCrop(pad_crop_length=480000, pad_position='left', crop_position='center')
        self.cluster = cluster

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        file = self.data[idx][0]
        
        language = self.classes[self.data[idx][1]]

        wav, _ = librosa.load(file, sr=8000)
        wav = torch.from_numpy(wav)
        
        if(self.data.shape[1] == 3):
            wav_duration = self.data[idx][2]
        else:
            wav_duration = -1

        # upsample 8k -> 16k
        wav = self.upsample(wav).unsqueeze(dim=0) 

        # Mixup
        mixup_wav = torch.zeros(1)
        mixup_language = torch.zeros(14)

        ######## Applying Mixup #########
        probability = 0.5
        if self.is_train:
            current_class = self.datacsv.iloc[idx, 1]
            # current_lang = self.datacsv.iloc[idx, 2]
            # current_dia = self.datacsv.iloc[idx, 3]
            if random.random() <= probability:

                if self.cluster == "across":
                    mix_class = random.choice(list(self.classes_set - set(current_class)))
                    mix_rows = self.datacsv.loc[self.datacsv['class'] == mix_class]["audiopath"].values
                    mixup_file = random.choice(mix_rows)
                    mixup_language = self.classes[mix_class]

                elif self.cluster == "within":
                    mix_class = current_class
                    mix_rows = self.datacsv.loc[self.datacsv['class'] == mix_class]["audiopath"].values
                    mixup_file = random.choice(mix_rows)
                    mixup_language = self.classes[mix_class]

                else:
                    # random mixing
                    mixup_idx = random.randint(0, self.data.shape[0]-1)
                    mixup_file = self.data[mixup_idx][0]
                    mixup_language = self.classes[self.data[mixup_idx][1]]

                mixup_wav, _ = librosa.load(mixup_file, sr=8000)
                mixup_wav = torch.from_numpy(mixup_wav)

                mixup_wav = self.upsample(mixup_wav).unsqueeze(dim=0)

                mixup_wav = self.train_transform(mixup_wav)
        ######## Done applying Mixup #########
            wav = self.train_transform(wav)
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
        
