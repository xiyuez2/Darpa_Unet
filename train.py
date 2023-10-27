import os
import glob
import argparse

import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
# from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from torch.utils.data import DataLoader

from datasets import MAPData
# from .transforms import make_transform
from models import SegmentationModel

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--train_data_dir', type=str, default='/projects/bbym/shared/all_patched_data/training')
parser.add_argument('--test_data_dir', type=str, default='/projects/bbym/shared/all_patched_data/validation')

parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--precision', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--project', type=str, default='DARPA')
# parser.add_argument('--name', type=str, default='DARPA_Unet')
parser.add_argument('--model', type=str, default='Unet')
parser.add_argument('--encoder', type=str, default='None')

parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--kfold', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--optimizer', type=str, default='adamp')
parser.add_argument('--scheduler', type=str, default='reducelr')
# parser.add_argument('--loss', type=str, default='ce')

# parser.add_argument('--crop_image_size', type=int, default=512)
# parser.add_argument('--ShiftScaleRotateMode', type=int, default=4)
# parser.add_argument('--ShiftScaleRotate', type=float, default=0.2)
# parser.add_argument('--HorizontalFlip', type=float, default=0.2)
parser.add_argument('--filp_rate', type=float, default=0.4)
parser.add_argument('--color_jitter_rate', type=float, default=0.2)
args = parser.parse_args()
args = parser.parse_args()

args.name = args.project + "_" + args.model

if __name__ == '__main__':
    pl.seed_everything(args.seed)
    print("cuda:", torch.cuda.is_available())
    for idx in [1]:
        wandb_logger = WandbLogger(project=args.project, group=args.name, name=f'{args.name}_fold{idx + 1:02d}')
        checkpoint_callback = ModelCheckpoint(
            monitor="val/jaccard_index_value",
            dirpath="checkpoints",
            filename=f"{args.name}_fold{idx + 1:02d}_" + "{val/jaccard_index_value:.4f}",
            save_top_k=3,
            mode="max",
            # save_weights_only=True
        )
        early_stop_callback = EarlyStopping(monitor="val/loss", min_delta=0.00, patience=50, verbose=True,
                                            mode="min")

        model = SegmentationModel(args)
    
        train_dataset = MAPData(data_path="/projects/bbym/shared/all_patched_data/training",type="poly",range=(0,30000), filp_rate = args.filp_rate, color_jitter_rate = args.color_jitter_rate)
        val_dataset = MAPData(data_path="/projects/bbym/shared/all_patched_data/validation",type="poly",range=(30000,36000),train=False)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,num_workers=args.num_workers)

        trainer = pl.Trainer(accelerator='gpu',
                             devices=args.gpus,
                             precision=args.precision,
                             max_epochs=args.epochs,
                             log_every_n_steps=1,
                             # strategy='ddp',
                             # num_sanity_val_steps=0,
                             # limit_train_batches=5,
                             # limit_val_batches=1,
                             logger=wandb_logger,
                             callbacks=[checkpoint_callback, early_stop_callback])

        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        # trainer.test(dataloaders=test_dataloader)
        wandb.finish()