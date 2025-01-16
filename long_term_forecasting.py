import torch
from torch import nn
from torch import optim

import os
import time
import warnings
import numpy as np

from models import TimesNet
from data_provider import data_provider

class Long_Term_Forecast():
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        model = TimesNet(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(args=self.args, flag=flag)
        return data_set, data_loader

    def _select_optimizer(self):
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.args.learning_rate)
        return optimizer

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def validation(self, valid_data, valid_loader, criterion):
        pass

    def train(self, setting):
        pass

    def test(self, setting, test=0):
        pass
    