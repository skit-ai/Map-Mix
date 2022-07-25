import os
from argparse import ArgumentParser

import torch
import torch.utils.data as data

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from config import LIDConfig

# SEED
SEED=100
pl.utilities.seed.seed_everything(SEED)
torch.manual_seed(SEED)

os.environ['WANDB_MODE'] = 'online'


from Datasets.datasetLID import LIDDataset, collate_fn
from Models.lightning import LightningModel


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
    parser.add_argument('--mixup_type', type=str, default=LIDConfig.mixup_type)
    parser.add_argument('--cluster', type=str, default=LIDConfig.cluster)
    parser.add_argument('--unfreeze_last_conv_layers', action='store_true')
    parser.add_argument('--noise_dataset_path', type=str, default=None)
    
    parser = pl.Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()
    print(f'Training Model on LID Dataset\n#Cores = {hparams.n_workers}\t#GPU = {hparams.gpu}')

    # Training, Validation and Testing Dataset
    ## Training Dataset
    train_set = LIDDataset(
        CSVPath = hparams.train_path,
        hparams = hparams,
        is_train=True,
    )
    ## Training DataLoader
    trainloader = data.DataLoader(
        train_set, 
        batch_size=hparams.batch_size, 
        shuffle=True, 
        num_workers=hparams.n_workers,
        collate_fn = collate_fn,
    )
    ## Validation Dataset
    valid_set = LIDDataset(
        CSVPath = hparams.val_path,
        hparams = hparams,
        is_train=False
    )
    ## Validation Dataloader
    valloader = data.DataLoader(
        valid_set, 
        batch_size=hparams.batch_size,
        # hparams.batch_size, 
        shuffle=False, 
        num_workers=hparams.n_workers,
        collate_fn = collate_fn,
    )
    ## Testing Dataset
    test_set = LIDDataset(
        CSVPath = hparams.test_path,
        hparams = hparams,
        is_train=False
    )
    ## Testing Dataloader
    testloader = data.DataLoader(
        test_set, 
        batch_size=hparams.batch_size,
        shuffle=False, 
        num_workers=hparams.n_workers,
        collate_fn = collate_fn,
    )

    print('Dataset Split (Train, Validation, Test)=', len(train_set), len(valid_set), len(test_set))

    logger = WandbLogger(
        name=LIDConfig.run_name,
        project='LangID'
    )
    
    HPARAMS= vars(hparams)
    model = LightningModel(HPARAMS)


    model_checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        monitor='val/loss', 
        mode='min',
        verbose=1)

    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = Trainer(
        fast_dev_run=False, 
        gpus=hparams.gpu, 
        max_epochs=hparams.epochs, 
        checkpoint_callback=True,
        callbacks=[
            EarlyStopping(
                monitor='val/loss',
                min_delta=0.00,
                patience=50,
                verbose=True,
                mode='min'
                ),
            model_checkpoint_callback,
            lr_monitor,
        ],
        logger=logger,
        # resume_from_checkpoint=hparams.model_checkpoint,
        # distributed_backend='ddp'
        )

    trainer.fit(model, train_dataloader=trainloader, val_dataloaders=valloader)

    print('\n\nCompleted Training...\nTesting the model with checkpoint -', model_checkpoint_callback.best_model_path)