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
        self.params = params
        self.id = id
        self.r_bit = None         # SU / PU 별로 상이함

    def size_to_RB(self, size, se):
        # return size(bits -> number of required RBs)
        RB_bandwidth = self.params.s_subcarrier_bandwidth * self.params.s_n_subcarrier_RB           # 12 subcarriers (180 kHz)
        data_rate = RB_bandwidth * se       # in bps
        RB_bit = data_rate * 0.5            # bits / RB (0.5 ms)
        return ceil(size / RB_bit)


class PUE(User):
    def __init__(self, id, x, y, params):
        super().__init__(id, x, y, params)
        ### Communication settings
        self.tx_power = params.s_D2D_tx_power  # tx power in dBm scale
        ### Learning settings
        self.trainloader = None
        self.local_model = None
        self.local_optim = None
        self.s_local = None
        self.lr_scheduler = None
        self.DSI = None
        self.data_for_sending = self.params.s_model_size
        self.num_RBs = 0
        self.neighbors = []

    def set_trainloader(self, trainloader):
        self.trainloader = trainloader

    def set_DSI(self, DSI):
        self.DSI = DSI

    def data_sending_reset(self):
        self.data_for_sending = self.params.s_model_size


class CUE(User):
    def __init__(self, id, x, y, params):
        super().__init__(id, x, y, params)
        ### Communication settings
        self.tx_power = params.s_uplink_tx_power  # tx power in dBm scale
        self.num_RBs = None
        self.pair = None

    def set_RBs(self, num):
        self.num_RBs = num

