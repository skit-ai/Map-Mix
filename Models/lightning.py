import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import pytorch_lightning as pl
from torchmetrics import Accuracy
from torchmetrics import F1Score
# from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from Models.model import UpstreamTransformer, UpstreamTransformerXLSR
from utils import CrossEntropyLoss


class LightningModel(pl.LightningModule):
    def __init__(self, HPARAMS):
        super().__init__()
        # HPARAMS
        self.save_hyperparameters()
        self.model = UpstreamTransformerXLSR(upstream_model=HPARAMS['upstream_model'], feature_dim=HPARAMS['feature_dim'], unfreeze_last_conv_layers=HPARAMS['unfreeze_last_conv_layers'])
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

    def forward(self, x, mixup_x, x_len, mixup_x_len, apply_mixup, lam, mixup_type):
        return self.model(x, mixup_x, x_len, mixup_x_len, apply_mixup, lam, mixup_type)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=5, max_epochs=100)
        return [optimizer]
        # , [scheduler]

    def training_step(self, batch, batch_idx):
        x, mixup_x, y_l, mixup_y_l, x_len, mixup_x_len, filenames = batch
        y_l = torch.stack(y_l)
        mixup_y_l = torch.stack(mixup_y_l)
        if (not torch.equal(mixup_y_l.cpu(), torch.zeros(mixup_y_l.shape))):
            apply_mixup = True
            alpha = 1
            lam = np.random.beta(alpha, alpha)
            y_l = lam*y_l + (1-lam)*mixup_y_l

        y_hat_l = self(x, mixup_x, x_len, mixup_x_len, apply_mixup, lam, self.mixup_type)        
        probs = F.softmax(y_hat_l, dim=1)

        language_loss = self.classification_criterion(y_hat_l, y_l)

        winners = y_hat_l.argmax(dim=1)
        corrects = (winners == y_l.argmax(dim=1))
        language_acc = corrects.sum().float() / float( y_hat_l.size(0) )
        train_step_acc = self.accuracy_metric(y_hat_l.argmax(dim=1), y_l.argmax(dim=1))
        train_step_f1 = self.f1_metric(y_hat_l.argmax(dim=1), y_l.argmax(dim=1))
        loss = language_loss

        self.log("train/f1", train_step_acc, on_step=False, on_epoch=True)
        self.log("train/acc", train_step_f1, on_step=False, on_epoch=True)

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
        # self.log('train/acc',language_acc, on_step=False, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, x_len, y_l = batch
        y_l = torch.stack(y_l)

        y_hat_l = self(x, x_len)
        language_loss = self.classification_criterion(y_hat_l, y_l)

        winners = y_hat_l.argmax(dim=1)
        corrects = (winners == y_l.argmax(dim=1))
        language_acc = corrects.sum().float() / float( y_hat_l.size(0) )

        val_step_acc = self.accuracy_metric(y_hat_l.argmax(dim=1), y_l.argmax(dim=1))
        val_step_f1 = self.f1_metric(y_hat_l.argmax(dim=1), y_l.argmax(dim=1))

        self.log("val/f1", val_step_acc, on_step=False, on_epoch=True)
        self.log("val/acc", val_step_f1, on_step=False, on_epoch=True)

        loss = language_loss

        return {'val_loss':loss, 
                'val_language_acc':language_acc,
                }

    def validation_epoch_end(self, outputs):
        val_loss = torch.tensor([x['val_loss'] for x in outputs]).mean()
        language_acc = torch.tensor([x['val_language_acc'] for x in outputs]).mean()
        
        self.log('val/loss' , val_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log('val/acc',language_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, x_len, y_l  = batch
        y_l = torch.stack(y_l)

        y_hat_l = self(x, x_len)

        winners = y_hat_l.argmax(dim=1)
        corrects = (winners == y_l.argmax(dim=1))
        language_acc = corrects.sum().float() / float( y_hat_l.size(0) )

        return {'language_acc':language_acc}

    def test_epoch_end(self, outputs):
        language_acc = torch.tensor([x['language_acc'] for x in outputs]).mean()

        pbar = {'language_acc' : language_acc}
        self.logger.log_hyperparams(pbar)
        self.log_dict(pbar)