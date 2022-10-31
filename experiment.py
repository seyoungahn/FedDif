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
        self.record3_name = self.exp_name + "_record3"
        self.record1_fieldnames = ['comm_round', 'accuracy', 'loss', 'num_of_diff_rounds']
        self.record2_fieldnames = ['comm_round', 'diff_round', 'diff_effi', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.record3_fieldnames = ['comm_round', 'diff_round', 'RBs', 'sub-frame']

        self.device = params.t_gpu_no if torch.cuda.is_available() else 'cpu'
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

        for i in range(self.params.n_cusers):
            x, y = self.create_coordination()
            cue = CUE(i, x, y, self.params)
            self.server.append_CUE(cue)

        for i in range(self.params.n_users):
            print(self.server.PUE_list[i].DSI)

        for i in range(self.params.n_users):
            for j in range(self.params.n_users):
                if i == j:
                    continue
                self.server.PUE_list[i].neighbors.append(j)

    # Polar coordination
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

            num_RBs, num_subframes = self.server.calc_num_sub_frames_from_BS_to_UE()
            record3_data = [comm_round + 1, 'b', num_RBs, num_subframes]
            utils.write_csv(self.save_dir, self.record3_name, record3_data, self.record3_fieldnames)

            ### Diffusion start
            while True:
                logging.info("COMM ROUND: {} | DIFF ROUND: {}".format(comm_round + 1, diff_round + 1))
                IID_dist, DE, num_RBs, num_subframes = self.server.diffusion()
                if IID_dist is None:
                    break
                record2_data = [comm_round + 1, diff_round + 1, DE] + IID_dist
                utils.write_csv(self.save_dir, self.record2_name, record2_data, self.record2_fieldnames)
                record3_data = [comm_round + 1, diff_round + 1, num_RBs, num_subframes]
                utils.write_csv(self.save_dir, self.record3_name, record3_data, self.record3_fieldnames)
                diff_round += 1
                lr = (self.params.t_learning_rate / 2.) * (1 + np.cos((diff_round * np.pi)/20.0))
                logging.info("Learning rate: {}".format(lr))
                self.server.local_training(lr)
                self.server.shuffle()

            ### global aggregation
            print("GLOBAL AGGREGATION: ", end='')
            self.server.global_aggregation()
            global_acc, global_loss = self.server.evaluate()
            record1_data = [comm_round + 1, global_acc, global_loss, diff_round]
            utils.write_csv(self.save_dir, self.record1_name, record1_data, self.record1_fieldnames)
            logging.info("LEARNING PERFORMANCE - acc: {}, loss: {}".format(global_acc, global_loss))

            num_RBs, num_subframes = self.server.calc_num_sub_frames_from_UE_to_BS()
            record3_data = [comm_round + 1, 'a', num_RBs, num_subframes]
            utils.write_csv(self.save_dir, self.record3_name, record3_data, self.record3_fieldnames)

    def FedDif_weight_difference(self):
        ## Set IID dataset
        logging.info("+ Loading the IID dataset ...".format(self.params.t_dataset_type, self.params.alpha))
        trainloader_IID, testloader_IID = datautils.fetch_dataloader(self.params)
        print("Data_size: {}".format(len(trainloader_IID.dataset)))
        logging.info(" done.")

        ## Set data sizes
        data_sizes = []
        for i in range(self.params.n_users):
            data_sizes.append(len(self.server.PUE_list[i].trainloader.dataset))
        data_sizes = np.array(data_sizes)

        ## Set initial configurations for FL
        models = []
        optimizers = []
        global_model_FL = cnn.CNN(self.params).to(self.device) if self.params.t_cuda else cnn.CNN(self.params)
        # Same initializations to the global model of FedDif
        global_model_FL.load_state_dict(self.server.global_model.state_dict())
        for i in range(self.params.n_users):
            model = cnn.CNN(self.params).to(self.device) if self.params.t_cuda else cnn.CNN(self.params)
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
            models.append(model)
            optimizers.append(optimizer)

        ## Set initial configurations for centralized learning
        centralized_model = cnn.CNN(self.params).to(self.device) if self.params.t_cuda else cnn.CNN(self.params)
        # Same initializations to the global model of FedDif
        centralized_model.load_state_dict(self.server.global_model.state_dict())
        centralized_optim = optim.SGD(centralized_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

        for comm_round in range(self.params.t_num_rounds):
            logging.info("COMM ROUND: {}".format(comm_round + 1))
            for pue in self.server.PUE_list:
                temp_x, temp_y = self.create_coordination()
                pue.set_coordination(temp_x, temp_y)
            self.server.global_init(self.save_dir)
            lr = self.params.t_learning_rate
            logging.info("Learning rate: {}".format(lr))
            ## FedDif
            self.server.local_training(lr)
            diff_round = 0

            # self.server.init_grad()

            self.server.global_train_mode()
            ### Diffusion start
            while True:
                logging.info("COMM ROUND: {} | DIFF ROUND: {}".format(comm_round + 1, diff_round + 1))
                IID_dist, DE,  = self.server.diffusion()
                if IID_dist is None:
                    break
                record2_data = [comm_round + 1, diff_round + 1, DE] + IID_dist
                print(record2_data)
                utils.write_csv(self.save_dir, self.record2_name, record2_data, self.record2_fieldnames)
                diff_round += 1
                lr = lr * (diff_round / (diff_round + 1))
                logging.info("Learning rate: {}".format(lr))
                self.server.local_training(lr)

            ### global aggregation
            print("GLOBAL AGGREGATION: ", end='')
            self.server.global_aggregation()
            global_FedDif_acc, global_FedDif_loss = self.server.evaluate()
            record1_data = [comm_round + 1, global_FedDif_acc, global_FedDif_loss, diff_round]
            utils.write_csv(self.save_dir, self.record1_name, record1_data, self.record1_fieldnames)

            ## Federated learning
            logging.info("+ FEDERATED LEARNING")
            for i in range(self.params.n_users):
                models[i].load_state_dict(global_model_FL.state_dict())
                logging.info("  - LOCAL TRAINING (user #{})...".format(i + 1))
                train_metrics = train(models[i], optimizers[i], self.params, utils.loss_function, self.server.PUE_list[i].trainloader, utils.metrics, self.params)

            logging.info("  - GLOBAL AGGREGATION")

            # Global aggregation (Federated learning)
            global_model_dict = dict(global_model_FL.state_dict())
            aggregated_dict = dict(global_model_FL.state_dict())
            parties_dict = {}
            for i in range(len(models)):
                parties_dict[i] = dict(models[i].state_dict())
            beta = data_sizes / sum(data_sizes)
            print("Data_sizes: ", data_sizes)
            print("Sum: {}".format(sum(data_sizes)))
            print("Beta: ", beta)
            for name, param in global_model_dict.items():
                aggregated_dict[name].data.copy_(sum([beta[i] * parties_dict[i][name].data for i in range(len(models))]))
            global_model_FL.load_state_dict(aggregated_dict)
            global_valid_metrics = evaluate(global_model_FL, utils.loss_function, self.server.testloader, utils.metrics, self.params)
            global_FedAvg_acc = global_valid_metrics['accuracy']
            global_FedAvg_loss = global_valid_metrics['loss']

            ## Centralized learning
            logging.info("+ CENTRALIZED LEARNING")
            train_metrics = train(centralized_model, centralized_optim, self.params, utils.loss_function, trainloader_IID, utils.metrics, self.params)
            centralized_valid_metrics = evaluate(centralized_model, utils.loss_function, testloader_IID, utils.metrics, self.params)
            centralized_acc = centralized_valid_metrics['accuracy']
            centralized_loss = centralized_valid_metrics['loss']

            ## Weight difference
            wd_FedDif = weight_difference(self.server.global_model, centralized_model)
            wd_FedAvg = weight_difference(global_model_FL, centralized_model)
            we_FedDif = weight_error(self.server.global_model, centralized_model)
            we_FedAvg = weight_error(global_model_FL, centralized_model)
            logging.info("Communication rounds: {}".format(comm_round + 1))
            logging.info("           Accuracy \t\tLoss\t\tWD\t\t\tWE")
            logging.info("[FedDif]   {:02.4f} %\t\t{:02.4f}\t\t{:02.4f}\t\t{:02.4f}".format(global_FedDif_acc*100., global_FedDif_loss, wd_FedDif, we_FedDif))
            logging.info("[FedAvg]   {:02.4f} %\t\t{:02.4f}\t\t{:02.4f}\t\t{:02.4f}".format(global_FedAvg_acc*100., global_FedAvg_loss, wd_FedAvg, we_FedAvg))
            logging.info("[Baseline] {:02.4f} %\t\t{:02.4f}".format(centralized_acc*100., centralized_loss))
            utils.write_csv(self.save_dir, 'weight_difference',
                            [comm_round + 1,
                             global_FedDif_acc, global_FedAvg_acc, centralized_acc,
                             global_FedDif_loss, global_FedAvg_loss, centralized_loss,
                             wd_FedDif.item(), wd_FedAvg.item(), we_FedDif.item(), we_FedAvg.item()],
                            ['Epochs', 'Acc(FedDif)', 'Acc(FedAvg)', 'Acc(Centralized)', 'Loss(FedAvg)', 'Loss(FedDif)', 'Loss(Centralized)',
                             'Weight difference(FedDif)', 'Weight difference(FedAvg)', 'Weight divergence(FedDif)', 'Weight divergence(FedAvg)'])