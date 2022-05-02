import os
import utils
import torch
import random
import logging
import datautils
import models.resnet as resnet
import methods.FedAvg
import torch.optim as optim
import csv
import math
import copy
import numpy as np

import matplotlib.pyplot as plt

from server import *
from user import *

class Experiment:
    def __init__(self, exp_name, params):
        self.model_dir = os.path.join("experiments", exp_name)
        self.save_dir = os.path.join("save", exp_name)
        if not os.path.isdir(self.model_dir):
            utils.mkdir_p(self.model_dir)
        if not os.path.isdir(self.save_dir):
            utils.mkdir_p(self.save_dir)
        self.params = params
        self.exp_name = exp_name
        self.record1_name = self.exp_name + "_record1"
        self.record2_name = self.exp_name + "_record2"
        self.record1_fieldnames = ['comm_round', 'accuracy', 'loss', 'num_of_diff_rounds']
        self.record2_fieldnames = ['comm_round', 'diff_round', 'diff_effi', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print("DEVICE: {}".format(self.device))

        """ Component initializations """
        self.server = Server(self.params, self.model_dir, self.save_dir)
        self.server.init_FL_task(self.device)

        # Set self.trainloaders, self.testloaders, self.DSI
        if self.params.t_stat_het == 'iid':
            self.set_iid_dataset()
        elif self.params.t_stat_het == 'noniid':
            self.set_noniid_dataset()
        elif self.params.t_stat_het == 'dirichlet':
            self.set_noniid_dirichlet_dataset()

        self.server.set_testloader(self.testloader)

        for i in range(self.params.n_users):
            x, y = self.create_coordination()
            pue = PUE(i, x, y, self.params)
            pue.set_trainloader(self.trainloader[i])
            pue.set_DSI(self.DSI_list[i])
            self.server.append_PUE(pue)

        for i in range(self.params.n_users):
            print(self.server.PUE_list[i].DSI)

        for i in range(self.params.n_users):
            for j in range(self.params.n_users):
                if i == j:
                    continue
                self.server.PUE_list[i].neighbors.append(j)

    # 극좌표 기반 좌표 생성 함수
    def create_coordination(self):
        angle = random.random() * math.pi * 2
        radius = random.uniform(0.1, 1) * self.params.s_r_cell
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        return x, y

    """
    Dataset setters
    1. Classwise non-IID dataset
    2. non-IID dataset by shards
    3. IID dataset
    """
    def set_classwise_dataset(self):
        utils.set_logger(os.path.join(self.model_dir, 'train.log'))
        logging.info("+ Loading the datasets ({})...".format(self.params.t_dataset_type))

        trainloaders, testloader, DSI_list = datautils.fetch_noniid_classwise_dataloader(self.params)

        logging.info("- done.")
        self.trainloader = trainloaders
        self.testloader = testloader
        self.DSI_list = DSI_list

    def set_noniid_dataset(self):
        utils.set_logger(os.path.join(self.model_dir, 'train.log'))
        logging.info("+ Loading the non-IID datasets ({})...".format(self.params.t_dataset_type))

        trainloaders, testloader, DSI_list = datautils.fetch_noniid_dataloader(self.params)

        logging.info("- done.")
        self.trainloader = trainloaders
        self.testloader = testloader
        self.DSI_list = DSI_list

    def set_noniid_dirichlet_dataset(self):
        utils.set_logger(os.path.join(self.model_dir, 'train.log'))
        logging.info("+ Loading the non-IID datasets based on Dirichlet distributions ({}, alpha={})...".format(self.params.t_dataset_type, self.params.alpha))

        trainloaders, testloader, DSI_list = datautils.fetch_noniid_dirichlet_dataloader(self.params)

        logging.info("- done.")
        self.trainloader = trainloaders
        self.testloader = testloader
        self.DSI_list = DSI_list

    def set_iid_dataset(self):
        utils.set_logger(os.path.join(self.model_dir, 'train.log'))
        logging.info("+ Loading the IID datasets ({})...".format(self.params.t_dataset_type))

        trainloaders, testloader, DSI_list = datautils.fetch_iid_dataloader(self.params)

        logging.info("- done.")
        self.trainloader = trainloaders
        self.testloader = testloader
        self.DSI_list = DSI_list

    def init_xy_set(self, x, y, r):
        xs, ys = [], []
        for theta in range(0, 360):
            xs.append(x + r * math.cos(math.radians(theta)))
            ys.append(y + r * math.sin(math.radians(theta)))
        return xs, ys

    def set_neighbor(self, PUE_list):
        for PUE_i in PUE_list:
            for PUE_j in PUE_list:
                if PUE_i is PUE_j:
                    continue
                if utils.Euclidean(PUE_i.x, PUE_i.y, PUE_j.x, PUE_j.y) < self.params.s_D2D_range ** 2:
                    PUE_i.append_neighbor(PUE_j)

    def FedDif(self):
        for comm_round in range(self.params.t_num_rounds):
            logging.info("COMM ROUND: {}".format(comm_round + 1))
            for pue in self.server.PUE_list:
                temp_x, temp_y = self.create_coordination()
                pue.set_coordination(temp_x, temp_y)
            self.server.global_init(self.save_dir)
            lr = self.params.t_learning_rate
            logging.info("Learning rate: {}".format(lr))
            self.server.local_training(lr)
            diff_round = 0
            while True: # Diffusion start
                logging.info("COMM ROUND: {} | DIFF ROUND: {}".format(comm_round + 1, diff_round + 1))
                IID_dist, DE = self.server.diffusion()
                if IID_dist is None:
                    break
                record2_data = [comm_round + 1, diff_round + 1, DE] + IID_dist
                utils.write_csv(self.save_dir, self.record2_name, record2_data, self.record2_fieldnames)
                diff_round += 1
                lr = lr * (diff_round / (diff_round + 1))
                logging.info("Learning rate: {}".format(lr))
                self.server.local_training(lr)
                ## print
            # global aggregation
            print("GLOBAL AGGREGATION: ", end='')
            self.server.global_aggregation()
            global_acc, global_loss = self.server.evaluate()
            record1_data = [comm_round + 1, global_acc, global_loss, diff_round]
            utils.write_csv(self.save_dir, self.record1_name, record1_data, self.record1_fieldnames)
            logging.info("LEARNING PERFORMANCE - acc: {}, loss: {}".format(global_acc, global_loss))
            ## diffusion round 수 (cnt출력)