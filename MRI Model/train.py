import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import os
import argparse
import numpy as np
import pandas as pd
import hyperparameters as p
from sklearn.model_selection import train_test_split

from model_api import Model
from data import MRIDataset

args = None

def main():
    df = pd.read_csv('./data/cleaned_csv.csv')
    train_df, val_df = train_test_split(df, stratify=df['Diagnosis'], test_size=.1)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_ds = MRIDataset(train_df, transform=transform)
    val_ds = MRIDataset(val_df, transform=transform)
    train_dl = DataLoader(train_ds, batch_size=p.BATCH_SIZE, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=p.BATCH_SIZE, num_workers=2)

    model = Model(train_dl, val_dl)
    model.train(p.EPOCHS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MRI Segmentation - Bill Bai')
    parser.add_argument('--epochs', type=int, default=20, help='Epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--learning_rate', type=int, default=2e-4, help='Learning rate')

    args = parser.parse_args()
    p.EPOCHS = args.epochs
    p.BATCH_SIZE = args.batch_size
    p.LR = args.learning_rate

    main()

