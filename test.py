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
from tqdm import tqdm

from datasets import MAPData, eval_MAPData
# from .transforms import make_transform
from models import SegmentationModel

parser = argparse.ArgumentParser()
#system setting
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--precision', type=int, default=32)

# model setting
parser.add_argument('--project', type=str, default='DARPA')
parser.add_argument('--model', type=str, default='Unet')
parser.add_argument('--ckpt', type=str, default='./checkpoints/DARPA_Unet_fold02_val/jaccard_index_value=0.9229.ckpt')

# dataloader setting
parser.add_argument('--map_data_dir', type=str, default='/projects/bbym/shared/data/cma/validation/')
parser.add_argument('--map_file', type=str, default="CA_AZ_Needles.tif")
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=8)

args = parser.parse_args()
args = parser.parse_args()

args.name = args.project + "_" + args.model
# with torch.no_grad():
def test_map(model, val_loader, args):
    model.eval()
    gen = tqdm(val_loader, leave=True)
    preds = []
    for i, batch in enumerate(gen, start=1):
        cur_preds = model.test_step(batch,i)
        for p in cur_preds.detach().cpu().numpy():
            preds.append(p)
    return preds


if __name__ == '__main__':
    pl.seed_everything(args.seed)
    print("cuda:", torch.cuda.is_available())
    # init model and data
    model = SegmentationModel.load_from_checkpoint(checkpoint_path = args.ckpt,args = args)
    val_dataset = eval_MAPData(data_path=args.map_data_dir,type="poly",range=(30000,36000),mapName = args.map_file)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,num_workers=args.num_workers)
    
    # pred loop and save res
    preds = test_map(model, val_loader, args)
    preds = np.array(preds)
    np.save("res",preds)
    
    # cal metrics
    metrics = val_dataset.metrics(preds)
    print(metrics)
    # return metrics