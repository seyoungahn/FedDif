import copy
import csv
import random

import utils
from node import *
import models.resnet as resnet
import models.alexnet as alexnet
import models.cnn as cnn
import models.lstm as lstm
import models.fcn as fcn
import models.lenet as lenet
import models.logistic as logistic
import torch
from model import *
import torch.optim as optim
import os
from math import *
import numpy as np
from methods.FedAvg import *
from munkres import Munkres, DISALLOWED, print_matrix
from torchsummary import summary as summary_
# import hungarian

class Vertex:
    def __init__(self, level, weight, profit, bound, include):
        self.level = level
        self.weight = weight
        self.profit = profit
        self.bound = bound
        self.include = include

    def __lt__(self, other):
        return self.bound < other.bound

class Server(Node):
    def __init__(self, params, model_dir, save_dir):
        super().__init__(0, 0, params)
        self.id = -1
        self.model_dir = model_dir
        self.save_dir = save_dir
        self.params = params
        self.exp_name = None
        self.tx_power = 43           # BS tx power in dBm scale
        self.testloader = None
        self.global_model = None
        self.idx = []
        self.models = []
        self.curr_policy = []
        self.PUE_list = []
        self.CUE_list = []
        self.s_global_model = self.params.s_model_size # bit
        self.n_model = ceil(self.params.n_users * self.params.r_model)
        self.mat_visited = [[False for _ in range(self.params.n_users)] for _ in range(self.n_model)]
        self.hungarian = Munkres()

    @staticmethod
    def poisson_process(lam, scale, max_val):
        n_arrival = 0
        t_arrival = 0
        t_slot = scale

        while True:
            # Get the next probability value from Uniform(0,1)
            p = random.random()

            # Plug it into the inverse of the CDF of Exponential(_lambda)
            t_inter_arrival = -math.log(1.0 - p) / lam

            # Add the inter-arrival time to the running sum
            t_arrival = t_arrival + t_inter_arrival

            if t_arrival > t_slot:
                break
            else:
                # Increment the number of arrival per unit time
                n_arrival = n_arrival + 1

        return min(n_arrival, max_val)

    def append_PUE(self, user):
        self.PUE_list.append(user)

    def append_CUE(self, user):
        self.CUE_list.append(user)

    def get_data_size(self):
        data_size = np.zeros(len(self.models))
        for i in range(len(self.models)):
            data_size[i] = self.models[i].subchain_datasize
        return data_size

    def init_FL_task(self, device):
        ### Model types: resnet34, resnet18, alexnet, lenet, cnn, lstm, fcn, logistic, svm
        if self.params.t_model_version == 'resnet34':
            self.global_model = resnet.ResNet34().to(device) if self.params.t_cuda else resnet.ResNet34()
            for i in range(self.n_model):
                model = Model(i, self.params)
                model.ML_model = resnet.ResNet34().to(device) if self.params.t_cuda else resnet.ResNet34()
                self.models.append(model)
        elif self.params.t_model_version == 'resnet18':
            self.global_model = resnet.ResNet18().to(device) if self.params.t_cuda else resnet.ResNet18()
            for i in range(self.n_model):
                model = Model(i, self.params)
                model.ML_model = resnet.ResNet18().to(device) if self.params.t_cuda else resnet.ResNet18()
                self.models.append(model)
        elif self.params.t_model_version == 'alexnet':
            self.global_model = alexnet.AlexNet().to(device) if self.params.t_cuda else alexnet.AlexNet()
            for i in range(self.n_model):
                model = Model(i, self.params)
                model.ML_model = alexnet.AlexNet().to(device) if self.params.t_cuda else alexnet.AlexNet()
                self.models.append(model)
        elif self.params.t_model_version == 'lenet':
            self.global_model = lenet.LeNet(self.params.t_class_num).to(device) if self.params.t_cuda else lenet.LeNet(self.params.t_class_num)
            # summary_(self.global_model, (3, 32, 32), batch_size=50)
            for i in range(self.n_model):
                model = Model(i, self.params)
                model.ML_model = lenet.LeNet(self.params.t_class_num).to(device) if self.params.t_cuda else lenet.LeNet(self.params.t_class_num)
                self.models.append(model)
        elif self.params.t_model_version == 'cnn':
            self.global_model = cnn.CNN(self.params).to(device) if self.params.t_cuda else cnn.CNN(self.params)
            for i in range(self.n_model):
                model = Model(i, self.params)
                model.ML_model = cnn.CNN(self.params).to(device) if self.params.t_cuda else cnn.CNN(self.params)
                self.models.append(model)
        elif self.params.t_model_version == 'lstm':
            self.global_model = lstm.LSTM(self.params).to(device) if self.params.t_cuda else lstm.LSTM(self.params)
            for i in range(self.n_model):
                model = Model(i, self.params)
                model.ML_model = lstm.LSTM(self.params).to(device) if self.params.t_cuda else lstm.LSTM(self.params)
                self.models.append(model)
        elif self.params.t_model_version == 'fcn':
            self.global_model = fcn.FCN(self.params.t_class_num).to(device) if self.params.t_cuda else fcn.FCN(self.params.t_class_num)
            for i in range(self.n_model):
                model = Model(i, self.params)
                model.ML_model = fcn.FCN(self.params.t_class_num).to(device) if self.params.t_cuda else fcn.FCN(self.params.t_class_num)
                self.models.append(model)
        elif self.params.t_model_version == 'logistic':
            self.global_model = logistic.Logistic().to(device) if self.params.t_cuda else logistic.Logistic()
            for i in range(self.n_model):
                model = Model(i, self.params)
                model.ML_model = logistic.Logistic().to(device) if self.params.t_cuda else logistic.Logistic()
                self.models.append(model)
        # elif self.params.t_model_version == 'svm':
        #     self.global_model = svm.SVM().to(device) if self.params.t_cuda else svm.SVM()
        #     for i in range(self.n_model):
        #         model = Model(i, self.params)
        #         model.ML_model = svm.SVM().to(device) if self.params.t_cuda else svm.SVM()
        #         self.models.append(model)

        self.n_par = len(utils.get_mdl_params([self.global_model])[0])
        self.state_gradient_diff = np.zeros(self.n_par)

        ### Save the parameters of the global model
        torch.save(self.global_model.state_dict(), os.path.join(self.save_dir, "global_model.pth"))
        self.s_global_model = os.path.getsize(os.path.join(self.save_dir, "global_model.pth"))

    # [Step 1] Global initialization
    def global_init(self):
        self.idx = list(range(0, self.params.n_users))
        random.shuffle(self.idx)
        self.idx = self.idx[:self.n_model]
        self.mat_visited = [[False for _ in range(self.params.n_users)] for _ in range(self.n_model)]
        self.curr_policy = [True for _ in range(self.n_model)]
        self.diffusion_efficiency = [0.0 for _ in range(self.n_model)]
        self.IID_dist_diffs = [0.0 for _ in range(self.n_model)]
        self.comm_cost = [0.0 for _ in range(self.n_model)]
        self.global_delta_sum = np.zeros(self.n_par)
        for i, model in enumerate(self.models):
            ## 1. Model initialization
            model.ML_model.load_state_dict(self.global_model.state_dict())
            model.global_model = self.global_model
            # model.curr_state_gradient_diff = np.zeros(self.n_par)
            # model.prev_state_gradient_diff = np.zeros(self.n_par)
            model.clear_config()
            ## 2. Model distribution (diffusion 시작 전)
            model.curr_trainer = self.idx[i]
            model.curr_DoL = self.PUE_list[self.idx[i]].DSI
            model.prev_trainer = self.idx[i]
            model.prev_DoL = self.PUE_list[self.idx[i]].DSI
            model.diffusion_subchain.append(self.idx[i])
            model.subchain_datasize = len(self.PUE_list[self.idx[i]].trainloader.dataset)
            self.visit(model.id, self.idx[i])


    def visit(self, model_id, PUE_id):
        self.mat_visited[model_id][PUE_id] = True

    def calc_num_sub_frames_from_BS_to_UE(self):
        for model in self.models:
            PUE = self.PUE_list[model.curr_trainer]
            PUE.data_sending_reset()
        num_RBs = 0
        num_sub_frames = 0
        while True:
            num_sub_frames += 1
            num_PUE_RB = self.poisson_process(1200, 1, 1484)
            list_PUE_dist = [random.random() + 0.2 for _ in range(self.params.n_cusers + self.n_model)]
            flag = True
            for i, model in enumerate(self.models):
                PUE = self.PUE_list[model.curr_trainer]
                num_PUE_RB_local = ceil(num_PUE_RB * list_PUE_dist[i] / sum(list_PUE_dist))
                datarate = 0.001 * utils.datarate(self.params, self, PUE, num_PUE_RB_local)
                if PUE.data_for_sending < datarate:
                    PUE.data_for_sending = 0
                    num_RBs += ceil(PUE.data_for_sending / (self.params.s_timeslot * self.params.s_subcarrier_bandwidth * self.params.s_n_subcarrier_RB * utils.spectral_efficiency_user(self.params, PUE, self, 0)))
                else:
                    PUE.data_for_sending -= datarate
                    num_RBs += num_PUE_RB_local
                    flag = False
            if flag is True:
                break
        return num_RBs, num_sub_frames

    def calc_num_sub_frames_from_UE_to_BS(self):
        for i in range(self.params.n_users):
            PUE = self.PUE_list[i]
            PUE.data_sending_reset()
        num_sub_frames = 0
        num_RBs = 0
        while True:
            num_sub_frames += 1
            num_PUE_RB = self.poisson_process(1200, 1, 1484)
            list_PUE_dist = [random.random() + 0.2 for _ in range(self.params.n_cusers + self.params.n_users)]
            flag = True
            for i in range(self.params.n_users):
                PUE = self.PUE_list[i]
                num_PUE_RB_local = ceil(num_PUE_RB * list_PUE_dist[i] / sum(list_PUE_dist))
                datarate = 0.001 * utils.datarate(self.params, PUE, self, num_PUE_RB_local)
                if PUE.data_for_sending < datarate:
                    PUE.data_for_sending = 0
                    num_RBs += ceil(PUE.data_for_sending / (self.params.s_timeslot * self.params.s_subcarrier_bandwidth * self.params.s_n_subcarrier_RB * utils.spectral_efficiency_user(self.params, PUE, self, 0)))
                else:
                    PUE.data_for_sending -= datarate
                    num_RBs += num_PUE_RB_local
                    flag = False
            if flag is True:
                break
        return num_RBs, num_sub_frames

    def scheduling(self, mat_value, mat_num_bits, mat_num_RBs):
        policy = [-1 for _ in range(self.n_model)]
        profit_matrix = [[0 for _ in range(self.params.n_users)] for _ in range(self.n_model)]
        for i in range(self.n_model):
            for j in range(self.params.n_users):
                if (mat_value[i][j] < 0) or (mat_num_bits[i][j] < 0):
                    profit_matrix[i][j] = 0
                else:
                    profit_matrix[i][j] = -ceil(mat_value[i][j] / mat_num_bits[i][j] * 8e13)
        results = self.hungarian.compute(profit_matrix)
        if results is False:
            return policy, 0, 0
        sum_RBs = 0
        for (model_idx, PUE_idx) in results:
            if profit_matrix[model_idx][PUE_idx] == 0:
                continue
            policy[model_idx] = PUE_idx
            sum_RBs += mat_num_RBs[model_idx][PUE_idx]
        num_sub_frames = 0
        transmitted_RBs = 0
        while transmitted_RBs < sum_RBs:
            num_sub_frames += 1
            transmitted_RBs += self.poisson_process(1200, 1, 1484)
        return policy, sum_RBs, num_sub_frames

    def diffusion(self):
        self.curr_policy = [False for _ in range(self.n_model)]

        # Step 2-1, 2-2: Bidding price calculation
        IID = np.array([1.0 / self.params.t_class_num for _ in range(self.params.t_class_num)])
        bidding_price = []
        for user in self.PUE_list:
            price_list = [0 for _ in range(self.n_model)]
            for model in self.models:
                if model.curr_trainer in user.neighbors:
                    prelim_DoL = model.get_next_DoL(user.DSI, len(user.trainloader.dataset))
                    valuation = utils.prob_dist(prelim_DoL, IID, self.params.t_dist)
                    price_list[model.id] = valuation
            bidding_price.append(price_list)

        # Step 2-3: Diffusion configuration
        value = [[0 for _ in range(self.params.n_users)] for _ in range(self.n_model)]  # N_model X N_PUE
        num_bits = [[0 for _ in range(self.params.n_users)] for _ in range(self.n_model)]  # N_model X N_PUE
        n_RBs = [[0 for _ in range(self.params.n_users)] for _ in range(self.n_model)]  # N_model X N_PUE
        for model in self.models:
            model.diffusion_round += 1
            from_user = self.PUE_list[model.curr_trainer]
            for user in self.PUE_list:
                user.data_sending_reset()
                if model.curr_trainer == user.id or self.mat_visited[model.id][user.id] is True:
                    value[model.id][user.id] = -math.inf
                    num_bits[model.id][user.id] = -math.inf
                else:
                    prev_IID_dist = utils.prob_dist(np.array(model.prev_DoL), IID, self.params.t_dist)
                    value[model.id][user.id] = prev_IID_dist - bidding_price[user.id][model.id]
                    n_RBs[model.id][user.id] = ceil(self.params.s_model_size / (self.params.s_timeslot * self.params.s_subcarrier_bandwidth * self.params.s_n_subcarrier_RB * utils.spectral_efficiency_user(self.params, from_user, user, 0)))
                    num_bits[model.id][user.id] = self.params.s_timeslot * self.params.s_subcarrier_bandwidth * self.params.s_n_subcarrier_RB * n_RBs[model.id][user.id]
        policy_model, num_RBs, num_sub_frames = self.scheduling(value, num_bits, n_RBs)
        for i in range(self.n_model):
            if policy_model[i] >= 0:
                self.curr_policy[i] = True
            else:
                self.curr_policy[i] = False
        DE = 0
        num_scheduled_model = 0
        IID_dist = []
        for i in range(self.n_model):
            if policy_model[i] < 0:
                IID_dist.append(-1)
            else:
                num_scheduled_model += 1
                self.IID_dist_diffs[i] = value[i][policy_model[i]]
                self.comm_cost[i] = num_bits[i][policy_model[i]]
                self.diffusion_efficiency[i] = self.IID_dist_diffs[i] / self.comm_cost[i] * 8e6
                DE += self.diffusion_efficiency[i]
                IID_dist.append(bidding_price[policy_model[i]][i])
        if num_scheduled_model > 0:
            DE /= self.n_model
        else:
            return None, None, None, None

        logging.info("Diffusion efficiency: {}".format(DE))

        if DE <= self.params.epsilon:
            return None, None, None, None

        # Step 2-4: Model transmission
        for i in range(self.n_model):
            if policy_model[i] != -1:
                self.visit(i, policy_model[i])
        for i in range(self.n_model):
            self.models[i].diffusion_subchain.append(policy_model[i])
            if policy_model[i] < 0:
                continue
            next_DoL = self.models[i].get_next_DoL(self.PUE_list[policy_model[i]].DSI, len(self.PUE_list[policy_model[i]].trainloader.dataset))
            self.models[i].prev_trainer = self.models[i].curr_trainer
            self.models[i].curr_trainer = policy_model[i]
            self.models[i].prev_DoL = self.models[i].curr_DoL
            self.models[i].curr_DoL = next_DoL
            self.models[i].subchain_datasize += len(self.PUE_list[policy_model[i]].trainloader.dataset)
        return IID_dist, DE, num_RBs, num_sub_frames

    def create_coordination(self):
        angle = random.random() * math.pi * 2
        radius = random.uniform(0.1, 1) * self.params.s_r_cell
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        return x, y

    def shuffle(self):
        for pue in self.PUE_list:
            temp_x, temp_y = self.create_coordination()
            pue.set_coordination(temp_x, temp_y)
        for cue in self.CUE_list:
            temp_x, temp_y = self.create_coordination()
            cue.set_coordination(temp_x, temp_y)

    def global_step(self):
        for i in range(self.n_model):
            self.models[i].optimizer.step()

    def global_train_mode(self):
        for i in range(self.n_model):
            self.models[i].ML_model.train()

    def random_diffusion(self):
        self.idx = list(range(0, self.params.n_users))
        random.shuffle(self.idx)
        self.idx = self.idx[:self.n_model]
        for i in range(self.n_model):
            self.models[i].prev_trainer = self.models[i].curr_trainer
            self.models[i].curr_trainer = self.idx[i]
            self.models[i].diffusion_subchain.append(self.idx[i])

    def local_training(self, lr, diff_rnd):
        acc_loss_record = np.zeros(5)
        cnt = 0
        for i in range(self.n_model):
            if self.curr_policy[i] is True:
                model = self.models[i]
                # model.optimizer.param_groups[0]['lr'] = lr
                n_iters = self.params.t_local_epochs * len(self.PUE_list[model.curr_trainer].trainloader)
                # if model.prev_trainer == model.curr_trainer:
                #     prev_update_last = self.state_gradient_diff
                # else:
                #     prev_update_last = self.PUE_list[model.prev_trainer].state_gradient_diff
                # global_update_last = self.PUE_list[model.prev_trainer].state_gradient_diff
                global_update_last = self.state_gradient_diff
                local_update_last = self.PUE_list[model.curr_trainer].state_gradient_diff

                # data_diff = model.curr_DoL - self.PUE_list[model.curr_trainer].DSI
                # data_diff = self.PUE_list[model.curr_trainer].DSI
                data_diff = model.curr_DoL

                _, loss_record, delta_param_curr = model.local_train(self.PUE_list[model.curr_trainer].trainloader, data_diff, lr, local_update_last, global_update_last, self.PUE_list[model.curr_trainer].parameter_drift, diff_rnd)

                self.PUE_list[model.curr_trainer].parameter_drift += delta_param_curr
                beta = 1 / n_iters / lr

                # self.delta_state_sum += model.delta_state_curr
                global_state = local_update_last - global_update_last + beta * (-delta_param_curr)
                global_delta_curr = global_state - self.PUE_list[model.curr_trainer].state_gradient_diff
                # global_delta_curr = global_state - self.state_gradient_diff
                # global_delta_curr = global_state - (self.PUE_list[model.curr_trainer].state_gradient_diff - self.state_gradient_diff)
                self.global_delta_sum += global_delta_curr
                self.PUE_list[model.curr_trainer].state_gradient_diff = global_state

                acc_loss_record += loss_record
                cnt += 1

        acc_loss_record /= cnt
        with open('reg_record_' + self.exp_name + '.csv', 'a') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(acc_loss_record)

    def global_aggregation(self, parameter_drifts):
        parameter_drifts_copy = copy.deepcopy(self.global_model)
        parameter_drifts_model = utils.set_client_from_params(parameter_drifts_copy, np.mean(parameter_drifts, axis=0), self.params.t_gpu_no)
        data_size = self.get_data_size()
        aggregated_dict = FedAvg(self.global_model, self.models, data_size, parameter_drifts_model)
        self.global_model.load_state_dict(aggregated_dict)
        self.save_checkpoint(self.save_dir)

    # def global_aggregation(self, parameter_drifts):
    #     parameter_drifts_copies = []
    #     for i, model in enumerate(self.models):
    #         parameter_drifts_copy = copy.deepcopy(model.ML_model)
    #         parameter_drifts_model = utils.set_client_from_params(parameter_drifts_copy, parameter_drifts[i], self.params.t_gpu_no)
    #         parameter_drifts_copies.append(parameter_drifts_model)
    #
    #     data_size = self.get_data_size()
    #     aggregated_dict = FedAvg(self.global_model, self.models, data_size, parameter_drifts_copies)
    #     self.global_model.load_state_dict(aggregated_dict)
    #     self.save_checkpoint(self.save_dir)

    def evaluate(self):
        logging.info("Global model evaluation")
        global_valid_metrics = evaluate(self.global_model, utils.loss_function, self.testloader, utils.metrics, self.params)
        global_acc_temp = global_valid_metrics['accuracy']
        global_loss_temp = global_valid_metrics['loss']
        return global_acc_temp, global_loss_temp

    def save_checkpoint(self, save_dir):
        ### Save the parameters of the global model
        torch.save(self.global_model.state_dict(), os.path.join(save_dir, "global_model.pth"))
        self.s_global_model = os.path.getsize(os.path.join(save_dir, "global_model.pth"))

    def set_testloader(self, testloader):
        self.testloader = testloader