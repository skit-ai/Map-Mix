import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from Models.model import UpstreamTransformer, UpstreamTransformerXLSR, PretrainedLangID
from utils import CrossEntropyLoss


class LightningModel(pl.LightningModule):
    def __init__(self, HPARAMS):
        super().__init__()
        # HPARAMS
        self.save_hyperparameters()
        self.models = {
            'UpstreamTransformer': UpstreamTransformer, # wav2vec, hubert
            'UpstreamTransformerXLSR': UpstreamTransformerXLSR, # XLSR
            'PretrainedLangID': PretrainedLangID,
        }
        self.model = self.models[HPARAMS['model_type']](upstream_model=HPARAMS['upstream_model'], feature_dim=HPARAMS['feature_dim'], unfreeze_last_conv_layers=HPARAMS['unfreeze_last_conv_layers'])
        self.classification_criterion = CrossEntropyLoss()
        self.lr = HPARAMS['lr']
        print(f"Model Details: #Params = {self.count_total_parameters()}\t#Trainable Params = {self.count_trainable_parameters()}")

    def count_total_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x, x_len):
        return self.model(x, x_len)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=5, max_epochs=100)
        return [optimizer]
        # , [scheduler]

    def training_step(self, batch, batch_idx):
        x, x_len, y_l = batch
        y_l = torch.stack(y_l)
        y_hat_l = self(x, x_len)        
        probs = F.softmax(y_hat_l, dim=1)

        loss = self.classification_criterion(y_hat_l, y_l)

        winners = y_hat_l.argmax(dim=1)
        corrects = (winners == y_l.argmax(dim=1))
        language_acc = corrects.sum().float() / float( y_hat_l.size(0) )

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
        self.log('train/acc',language_acc, on_step=False, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, x_len, y_l = batch
        y_l = torch.stack(y_l)

        y_hat_l = self(x, x_len)
        loss = self.classification_criterion(y_hat_l, y_l)

        winners = y_hat_l.argmax(dim=1)
        corrects = (winners == y_l.argmax(dim=1))
        language_acc = corrects.sum().float() / float( y_hat_l.size(0) )

        return {'val_loss':loss, 
                'val_language_acc':language_acc,
                }

    def validation_epoch_end(self, outputs):
        val_loss = torch.tensor([x['val_loss'] for x in outputs]).mean()
        language_acc = torch.tensor([x['val_language_acc'] for x in outputs]).mean()
        
        self.log('val/loss' , val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc',language_acc, on_step=False, on_epoch=True, prog_bar=True)