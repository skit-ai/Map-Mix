import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import pytorch_lightning as pl
from torchmetrics import Accuracy
from torchmetrics import F1Score
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

# from Models.model import UpstreamTransformerXLSR
from utils import CrossEntropyLoss
from transformers import Wav2Vec2Processor, Wav2Vec2Model


class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling 
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)
        
    def forward(self, batch_rep):
        """
        input:
            batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension
        
        attention_weight:
            att_w : size (N, T, 1)
        
        return:
            utter_rep: size (N, H)
        """
        softmax = nn.functional.softmax
        att_w = softmax(self.W(batch_rep).squeeze(-1)).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep

class LightningModel(pl.LightningModule):
    def __init__(self, HPARAMS):
        super().__init__()
        # HPARAMS
        # self.save_hyperparameters()

        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m")

        for param in self.encoder.parameters():
            param.requires_grad = True
        
        for param in self.encoder.encoder.layers.parameters():
            param.requires_grad = True

        self.attn_pool = SelfAttentionPooling(HPARAMS['feature_dim'])

        self.language_classifier = nn.Sequential(
            nn.Linear(HPARAMS['feature_dim'], 512),
            nn.ReLU(),
            nn.Linear(512, 14)
        )

        self.classification_criterion = CrossEntropyLoss()
        self.accuracy_metric = Accuracy()
        self.f1_metric = F1Score()
        self.lr = HPARAMS['lr']
        self.mixup_type = HPARAMS['mixup_type']

        print(f"Model Details: #Params = {self.count_total_parameters()}\t#Trainable Params = {self.count_trainable_parameters()}")

    def count_total_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def simple_forward(self, x, x_len):
        x = self.encoder(x).last_hidden_state
        x = self.attn_pool(x)
        language = self.language_classifier(x)
        return language

    def mixup_forward(self, x, mixup_x, x_len, mixup_x_len, lam):
        x = self.encoder(x).last_hidden_state
        mixup_x = self.encoder(mixup_x).last_hidden_state
        
        x = self.attn_pool(x)
        mixup_x = self.attn_pool(mixup_x)
        
        x = lam*x + (1-lam)*mixup_x

        language = self.language_classifier(x)
        return language

    def forward(self, x, mixup_x, x_len, mixup_x_len, apply_mixup, lam, mixup_type):
        x = self.processor(x, sampling_rate=16000, return_tensors="pt")["input_values"].squeeze(0).to(self.device)
        mixup_x = self.processor(mixup_x, sampling_rate=16000, return_tensors="pt")["input_values"].squeeze(0).to(self.device)
        if(apply_mixup):
            if mixup_type == 'latent-mixup-last':
                return self.mixup_forward(x, mixup_x, x_len, mixup_x_len, lam)
            elif mixup_type == 'static':
                x = lam*x + (1-lam)*mixup_x
                return self.simple_forward(x, x_len)
            else:
                raise Exception("Wrong mixup type. Supported types - 1. latent-mixup-last 2. static")
        else:
            return self.simple_forward(x, x_len)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return [optimizer]

    def training_step(self, batch, batch_idx):
        x, mixup_x, y_l, mixup_y_l, x_len, mixup_x_len, filenames = batch
        y_l = torch.stack(y_l)
        mixup_y_l = torch.stack(mixup_y_l)

        apply_mixup = True
        alpha = 0.5 # updated based on the original paper
        lam = np.random.beta(alpha, alpha)
        y_l = lam*y_l + (1-lam)*mixup_y_l

        y_hat_l = self(x, mixup_x, x_len, mixup_x_len, apply_mixup, lam, self.mixup_type)        
        probs = F.softmax(y_hat_l, dim=1)

        language_loss = self.classification_criterion(y_hat_l, y_l)

        winners = y_hat_l.argmax(dim=1)
        corrects = (winners == y_l.argmax(dim=1))
        language_acc = corrects.sum().float() / float( y_hat_l.size(0) )
        train_step_acc = self.accuracy_metric(y_hat_l.argmax(dim=1), y_l.argmax(dim=1))
        loss = language_loss
        self.log("train/acc", train_step_acc, on_step=False, on_epoch=True)

        return {'loss':loss, 
                'language_acc':language_acc,
                'probs': probs.detach().cpu().numpy(),
                'labels': y_l.argmax(dim=1).detach().cpu().numpy().astype(int),
                }
    
    def training_epoch_end(self, outputs):
        n_batch = len(outputs)
        loss = torch.tensor([x['loss'] for x in outputs]).mean()
        language_acc = torch.tensor([x['language_acc'] for x in outputs]).mean()

        self.log('train/loss' , loss, on_step=False, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, x_len, y_l = batch
        y_l = torch.stack(y_l)

        y_hat_l = self.simple_forward(x, x_len)
        language_loss = self.classification_criterion(y_hat_l, y_l)

        winners = y_hat_l.argmax(dim=1)
        corrects = (winners == y_l.argmax(dim=1))
        language_acc = corrects.sum().float() / float( y_hat_l.size(0) )

        val_step_acc = self.accuracy_metric(y_hat_l.argmax(dim=1), y_l.argmax(dim=1))
        self.log("val/acc", val_step_acc, on_step=False, on_epoch=True)

        loss = language_loss

        return {'val_loss':loss, 
                'val_language_acc':language_acc,
                }

    def validation_epoch_end(self, outputs):
        val_loss = torch.tensor([x['val_loss'] for x in outputs]).mean()
        language_acc = torch.tensor([x['val_language_acc'] for x in outputs]).mean()
        
        self.log('val/loss' , val_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log('val/acc',language_acc, on_step=False, on_epoch=True, prog_bar=True)