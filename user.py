import logging

import methods.FedAvg
import utils
from node import *

from math import ceil
import time

import models.resnet as resnet
import models.alexnet as alexnet
import models.cnn as cnn
import torch
import torch.optim as optim
import os


class User(Node):
    def __init__(self, id, x, y, params):
        super().__init__(x, y, params)
        self.id = id
        self.r_bit = None         # SU / PU 별로 상이함

    @staticmethod
    def size_to_RB(size, se):
        # return size(bits -> number of required RBs)
        RB_bandwidth = self.params.s_subcarrier_bandwidth * self.params.s_n_subcarrier_RB           # 12 subcarriers (180 kHz)
        data_rate = RB_bandwidth * se       # in bps
        RB_bit = data_rate * 0.5            # bits / RB (0.5 ms)
        return ceil(size / RB_bit)


class PUE(User):
    def __init__(self, id, x, y, params):
        super().__init__(id, x, y, params)
        ### Communication settings
        self.tx_power = params.s_D2D_tx_power  # tx power 별로 실험할 예정임 (in dBm scale)
        ### Learning settings
        self.trainloader = None
        self.local_model = None
        self.local_optim = None
        self.s_local = None
        self.lr_scheduler = None
        self.DSI = None
        self.neighbors = []

    def set_trainloader(self, trainloader):
        self.trainloader = trainloader

    def set_DSI(self, DSI):
        self.DSI = DSI
