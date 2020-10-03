import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

import os
import  time

import numpy as np

from models import UNet
import ml_utils as utils
from data import MRIDataset
from hyperparameters import *

class Model:
    def __init__(self, df):
        pass

    def _init_optim(self):
        pass

    def _save_models(self):
        pass

    def train(self):
        pass

    @classmethod
    def _init_weights(cls, layer: nn.Module):
        name = layer.__class__.__name__
        if name.find('Conv') != -1:
            nn.init.normal_(layer.weight.data, .0, 2e-2)
        if name.find('BatchNorm') != -1:
            nn.init.normal_(layer.weight.data, 1.0, 2e-2)
            nn.init.constant_(layer.bias.data, .0)

