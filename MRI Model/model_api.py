import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms

import os
import time

import numpy as np

import ml_utils as utils
from data import MRIDataset
from models import UNet, Loss
from hyperparameters import *


class Model:
    def __init__(self, train_dl, val_dl):
        self.device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.train_dl = train_dl
        self.val_dl = val_dl

        self.loss = Loss()
        self.net = UNet(1).to(self.device)
        self.net.apply(Model._init_weights)
        self.criterion = self.loss.BCEDiceLoss
        self.optim = None
        self.scheduler = None

        self._init_optim(LR, BETAS)

        self.cycles = 0
        self.hist = {
            'train': [],
            'val': [],
            'loss': []
        }

        utils.create_dir('./pt')
        utils.log_data_to_txt('train_log', f'\nUsing device {self.device}')

    def _init_optim(self, lr, betas):
        self.optim = optim.Adam(
            utils.filter_gradients(self.net)
            , lr=lr
        )

        self.scheduler = optim.lr_scheduler.StepLR(self.optim, step_size=100, gamma=.75)

    def _save_models(self):
        utils.save_state_dict(self.net, 'model', './pt')
        utils.save_state_dict(self.optim, 'optim', './pt')
        utils.save_state_dict(self.scheduler, 'scheduler', './pt')

    def train(self, epochs):
        self.net.train()
        for epoch in range(epochs):
            self.net.train()
            for idx, data in enumerate(self.train_dl):
                batch_time = time.time()

                self.cycles += 1
                print(self.cycles)

                image = data['MRI'].to(self.device)
                target = data['Mask'].to(self.device)

                output = self.net(image)

                output_rounded = np.copy(output.data.cpu().numpy())
                output_rounded[np.nonzero(output_rounded < 0.5)] = 0.
                output_rounded[np.nonzero(output_rounded >= 0.5)] = 1.
                train_f1 = self.loss.F1_metric(output_rounded, target.data.cpu().numpy())

                loss = self.criterion(output, target)

                self.hist['train'].append(train_f1)
                self.hist['loss'].append(loss.item())

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                self.scheduler.step()

                if self.cycles % 100 == 0:
                    self._save_models()
                    val_f1 = self.evaluate()
                    utils.log_data_to_txt('train_log',
                                          f'\nEpoch: {epoch}/{epochs} - Batch: {idx * BATCH_SIZE}/{len(self.train_dl.dataset)}'
                                          f'\nLoss: {loss.mean().item():.4f}'
                                          f'\nTrain F1: {train_f1:.4f} - Val F1: {val_f1}'
                                          f'\nTime taken: {time.time() - batch_time:.4f}s')

    def evaluate(self):
        # model.eval()
        loss_v = 0

        with torch.no_grad():
            for idx, data in enumerate(self.val_dl):
                image, target = data['MRI'], data['Mask']

                image = image.to(self.device)
                target = target.to(self.device)

                outputs = self.net(image)

                out_thresh = np.copy(outputs.data.cpu().numpy())
                out_thresh[np.nonzero(out_thresh < .3)] = 0.0
                out_thresh[np.nonzero(out_thresh >= .3)] = 1.0

                loss = self.loss.F1_metric(out_thresh, target.data.cpu().numpy())
                loss_v += loss

        return loss_v / idx

    @classmethod
    def _init_weights(cls, layer: nn.Module):
        name = layer.__class__.__name__
        if name.find('Conv') != -1 and name.find('2d') != -1:
            nn.init.normal_(layer.weight.data, .0, 2e-2)
        if name.find('BatchNorm') != -1:
            nn.init.normal_(layer.weight.data, 1.0, 2e-2)
            nn.init.constant_(layer.bias.data, .0)
