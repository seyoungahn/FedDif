import random
import queue

from node import *
import models.resnet as resnet
import models.alexnet as alexnet
import models.cnn as cnn
import models.lstm as lstm
import models.fcn as fcn
import torch
from model import *
import torch.optim as optim
import os
from math import *
import numpy as np
from methods.FedAvg import *

## TODO: 수형 추가 (branch-and-bound를 위한 class)
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
        self.tx_power = 43           # BS tx power 표준에서 구해야 함 in dBm scale
        self.testloader = None     # TODO: SY
        self.global_model = None    # TODO: SY
        self.idx = []
        self.models = []
        self.curr_policy = []
        self.PUE_list = []
        self.s_global_model = self.params.s_model_size      # bit
        self.n_model = ceil(self.params.n_users * self.params.r_model)
        self.mat_visited = [[False for _ in range(self.params.n_users)] for _ in range(self.n_model)]  ## TODO: 수형 추가함

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

    def get_data_size(self):
        data_size = np.zeros(len(self.models))
        for i in range(len(self.models)):
            data_size[i] = self.models[i].subchain_datasize
        return data_size

    def init_FL_task(self, device):
        ### Select the model
        if self.params.t_model_version == 'resnet18':
            self.global_model = resnet.ResNet18().to(device) if self.params.t_cuda else resnet.ResNet18()
            for i in range(self.n_model):
                model = Model(i, self.params)
                model.ML_model = resnet.ResNet18().to(device) if self.params.t_cuda else resnet.ResNet18()
                model.local_configuration()
                self.models.append(model)
        elif self.params.t_model_version == 'alexnet':
            self.global_model = alexnet.AlexNet().to(device) if self.params.t_cuda else alexnet.AlexNet()
            for i in range(self.n_model):
                model = Model(i, self.params)
                model.ML_model = alexnet.AlexNet().to(device) if self.params.t_cuda else alexnet.AlexNet()
                model.local_configuration()
                self.models.append(model)
        elif self.params.t_model_version == 'cnn':
            self.global_model = cnn.CNN().to(device) if self.params.t_cuda else cnn.CNN()
            for i in range(self.n_model):
                model = Model(i, self.params)
                model.ML_model = cnn.CNN().to(device) if self.params.t_cuda else cnn.CNN()
                model.local_configuration()
                self.models.append(model)
        elif self.params.t_model_version == 'lstm':
            self.global_model = lstm.LSTM().to(device) if self.params.t_cuda else lstm.LSTM()
            for i in range(self.n_model):
                model = Model(i, self.params)
                model.ML_model = lstm.LSTM().to(device) if self.params.t_cuda else lstm.LSTM()
                model.local_configuration()
                self.models.append(model)
        elif self.params.t_model_version == 'fcn':
            self.global_model = fcn.FCN().to(device) if self.params.t_cuda else fcn.FCN()
            for i in range(self.n_model):
                model = Model(i, self.params)
                model.ML_model = fcn.FCN().to(device) if self.params.t_cuda else fcn.FCN()
                model.local_configuration()
                self.models.append(model)

        ### Save the parameters of the global model
        torch.save(self.global_model.state_dict(), os.path.join(self.save_dir, "global_model.pth"))
        self.s_global_model = os.path.getsize(os.path.join(self.save_dir, "global_model.pth"))

    # [Step 1] Global initialization
    def global_init(self, save_dir):
        self.idx = list(range(0, self.params.n_users))
        random.shuffle(self.idx)
        self.mat_visited = [[False for _ in range(self.params.n_users)] for _ in range(self.n_model)]  ## TODO: 수형 추가함
        self.curr_policy = [True for _ in range(self.n_model)]
        # 1. Model initialization
        for model in self.models:
            model.ML_model.load_state_dict(self.global_model.state_dict())
            model.clear_model()
        # 2. Model distribution (diffusion 시작 전)
        for i in range(self.n_model):
            self.models[i].curr_trainer = self.idx[i]
            self.models[i].curr_DoL = self.PUE_list[self.idx[i]].DSI
            self.models[i].prev_trainer = self.idx[i]
            self.models[i].prev_DoL = self.PUE_list[self.idx[i]].DSI
            self.models[i].diffusion_subchain.append(self.idx[i])
            self.models[i].subchain_datasize = len(self.PUE_list[self.idx[i]].trainloader.dataset)
            self.visit(self.models[i].id, self.idx[i])  ## TODO: 수형 추가함 (diffusion chain 방문 현황 기록)

    ## TODO: 수형 추가함 (subchain 방문 기록용)
    def visit(self, model_id, PUE_id):
        self.mat_visited[model_id][PUE_id] = True

    ## TODO: 수형 추가함 (diffusion & resource allocation)
    def scheduling(self, cond, mat_value, mat_weight):
        policy = [-1 for _ in range(self.params.n_users)]

        scheduled_model = [False for _ in range(self.n_model)]
        scheduled_PUE = [False for _ in range(self.params.n_users)]

        list_val = []
        for i in range(self.n_model):
            for j in range(self.params.n_users):
                frac = 0
                if mat_value[i][j] > 0 and mat_weight[i][j] > 0:
                    frac = mat_value[i][j] / mat_weight[i][j]
                # (model idx, PUE idx, value, weight, frac)
                list_val.append((i, j, mat_value[i][j], mat_weight[i][j], frac))
        list_val.sort(key=lambda x: x[4], reverse=True)

        for val in list_val:
            idx_model, idx_PUE, val_value, val_weight, val_frac = val
            if scheduled_model[idx_model] is True or scheduled_PUE[idx_PUE] is True:
                continue
            policy[idx_model] = idx_PUE
            scheduled_model[idx_model] = True
            scheduled_PUE[idx_PUE] = True

        list_sort = []
        for i in range(self.n_model):
            frac = 0
            if mat_value[i][policy[i]] > 0 and mat_weight[i][policy[i]] > 0:
                frac = mat_value[i][policy[i]] / mat_weight[i][policy[i]]
            # (model idx, value, weight, frac)
            list_sort.append((i, mat_value[i][policy[i]], mat_weight[i][policy[i]], frac))
        list_sort.sort(key=lambda x: x[3], reverse=True)

        def calc_bound(node):
            if node.weight >= max_capacity:
                return False
            idx = node.level + 1
            bound = node.profit
            totweight = node.weight
            while idx < self.n_model and totweight + list_sort[idx][2] <= max_capacity:
                totweight += list_sort[idx][2]
                bound += list_sort[idx][1]
                idx += 1
            last = idx
            if last < self.n_model:
                bound += (max_capacity - totweight) * (list_sort[last][1] / list_sort[last][2])
            return bound

        scheduled_model = [False for _ in range(self.n_model)]
        scheduled_RB = []

        num_scheduled_model = 0
        idx_auction = 0
        while num_scheduled_model < cond:
            idx_auction += 1
            pq = queue.PriorityQueue()
            max_capacity = self.params.s_n_RB * self.params.s_n_timeslot
            for _ in range(self.params.s_n_timeslot):
                max_capacity -= self.poisson_process(25, 1, 50)  # TODO: auction 마다 num_RB random으로 바뀌어야 함
                                                                   # 평균, 도착수, 최댓값 # poisson process를 통해 도착한 CUE들은 평균 25개의 RB를 사용함.
            max_profit = 0
            knapsack_result = None
            temp = [False for _ in range(self.n_model)]

            level = 0
            while level < self.n_model:
                if scheduled_model[list_sort[level][0]] is True:
                    level += 1
                else:
                    break
            v = Vertex(-1, level, 0, 0.0, temp)
            v.bound = calc_bound(v)
            pq.put((-v.bound, v))

            while not pq.empty():
                v = pq.get()[1]
                if v.bound > max_profit:
                    level = v.level + 1
                    while level < self.n_model:
                        if scheduled_model[list_sort[level][0]] is True:
                            level += 1
                        else:
                            break
                    if level == self.n_model:
                        break
                    weight = v.weight + list_sort[level][2]
                    profit = v.profit + list_sort[level][1]
                    include = v.include[:]
                    u = Vertex(level, weight, profit, 0.0, include)
                    u.include[list_sort[level][0]] = True
                    if u.weight <= max_capacity and u.profit > max_profit:
                        max_profit = u.profit
                        knapsack_result = u.include
                    u.bound = calc_bound(u)
                    if u.bound > max_profit:
                        pq.put((-u.bound, u))

                    u = Vertex(level, v.weight, v.profit, 0.0, v.include)
                    u.bound = calc_bound(u)
                    if u.bound > max_profit:
                        pq.put((-u.bound, u))

            # TODO: 이게 진짜 맞는가?
            # -> knapsack_result is None 이라는 뜻은 얻을 수 있는 profit이 전부 음수라는 뜻임
            # -> 만약 rho * self.n_model 보다 적게 할당되었다고 하더라도 멈추는게 맞음 (?)
            if knapsack_result is None:
                break

            # Resource allocation
            resource_allocation = [-1 for _ in range(self.params.n_users)]
            for idx in range(self.n_model):
                if knapsack_result[idx] is True:
                    resource_allocation[self.models[idx].curr_trainer] = mat_weight[idx][policy[idx]]
            scheduled_RB.append(resource_allocation)

            for idx in range(self.n_model):
                if knapsack_result[idx] is True:
                    scheduled_model[idx] = True
                    num_scheduled_model += 1

        for idx in range(self.n_model):
            if scheduled_model[idx] is False:
                policy[idx] = -1

        return policy, idx_auction, scheduled_RB

    def diffusion(self):
        # # [Step 2-1] DoL broadcasting
        # prelim_DoL_list = []
        # for model in self.models:
        #     prelim_DoL_list.append(model.DoL_broadcasting(self.PUE_list))
        # # [Step 2-2] Preliminary IID distance reporting
        # IID = np.array([1.0 / self.params.t_class_num for _ in range(self.params.t_class_num)])
        # prelim_IID_dist_list = []
        # for model in prelim_DoL_list:
        #     prelim_IID_dist = []
        #     for prelim_DoL in model:
        #         prelim_IID_dist.append(np.linalg.norm(prelim_DoL - IID))
        #     prelim_IID_dist_list.append(prelim_IID_dist)
        self.curr_policy = [False for _ in range(self.n_model)]
        ## TODO: 여기서부터 제가 작성한 코드
        ## 제 생각에는 DoL broadcasting을 실제로 구현할 필요는 없을 것 같습니다.
        ## 각 PUE는 model의 curr_trainer가 내 neighbor라면 broadcasting 되었다고 생각하고 prelim_DoL을 계산합니다.
        ## prelim_DoL은 model.calc_DoL(PUE)로 함수를 만들었습니다.
        ## 기존 prelim_IID_dist_list -> bidding_price 로 바뀌었습니다.
        ## bidding_price = [[UE1->model1에 대한 value, UE1->model2에 대한 value, ...,  UE1->modelN에 대한 value], [UE2], ... [마지막 UE]]
        # TODO: neighbors에 PUE들을 넣는 작업 필요
        
        # [Step 2-1 and 2-2] bidding price calculation
        IID = np.array([1.0 / self.params.t_class_num for _ in range(self.params.t_class_num)])
        bidding_price = []
        for user in self.PUE_list:
            price_list = [0 for _ in range(self.n_model)]
            for model in self.models:
                if model.curr_trainer in user.neighbors:
                    # prelim_DoL = model.calc_DoL(user, self.PUE_list)
                    prelim_DoL = model.get_next_DoL(user.DSI, len(user.trainloader.dataset))
                    valuation = np.linalg.norm(prelim_DoL - IID, ord=2)
                    price_list[model.id] = valuation
            bidding_price.append(price_list)

        # [Step 2-3] Diffusion configuration
        ## TODO: Auction기반 scheduling
        value = [[0 for _ in range(self.params.n_users)] for _ in range(self.n_model)]  # N_model X N_PUE
        weight = [[0 for _ in range(self.params.n_users)] for _ in range(self.n_model)]  # N_model X N_PUE
        for model in self.models:
            model.diffusion_round += 1
            for user in self.PUE_list:
                if model.curr_trainer == user.id or self.mat_visited[model.id][user.id] is True:
                    value[model.id][user.id] = -math.inf
                    weight[model.id][user.id] = math.inf
                else:
                    prev_IID_dist = np.linalg.norm(np.array(model.prev_DoL) - IID, ord=2)
                    value[model.id][user.id] = prev_IID_dist - bidding_price[user.id][model.id]
                    weight[model.id][user.id] = self.PUE_list[model.curr_trainer].calc_num_RB(user)
        # 0-1 knapsack algorithm and resource allocation
        policy_model, num_auction, policy_RB = self.scheduling(self.params.rho * self.params.n_users, value, weight)

        for i in range(self.n_model):
            if policy_model[i] >= 0:
                self.curr_policy[i] = True
            else:
                self.curr_policy[i] = False
        ## TODO: 결과 설명
        ## policy_model = [idx1, idx2, ..., idx_n_model]
        ##    -> 각 모델이 다음에 어떤 PUE를 선택할지에 대한 idx를 반환함
        ##    -> idx가 -1일 수 있음: rho * n_users 이상 만큼 할당했을 때 할당받지 못한 model의 경우 -1로 표시함
        ##    -> rho는 params.rho = 0.8 로 정의했음
        ## num_auction = auction을 진행한 횟수
        ## policy_RB = [[num_RB1, num_RB2, ..., num_RB_n_users], [...], ...]
        ##    -> 안쪽 []: 매 auction 마다 각 PUE가 model을 보낼 때 필요한 RB 수를 반환함 (할당 X인경우 0)
        ##    -> []의 수 = num_auction 수

        ## TODO: 수형 추가함 (diffusion stop condition) diffusion efficiency update
        DE = 0
        num_scheduled_model = 0
        IID_dist = []
        for i in range(self.n_model):
            if policy_model[i] < 0:
                IID_dist.append(-1)
            else:
                num_scheduled_model += 1
                DE += value[i][policy_model[i]] / weight[i][policy_model[i]]
                IID_dist.append(bidding_price[policy_model[i]][i])
        if num_scheduled_model > 0:
            DE /= num_scheduled_model

        # diffusion stop condition (전체를 다 계산해봐야 하기 때문에 auction 전이 아니라 여기서 확인해야 함(구현의 편의를 위해))
        # model transmission 전에 수행해야 함
        ## TODO: stop condition이 이게 맞는가?
        # -> 구현해보니까 DE가 별로 변하지 않는다고 하더라도 epsilon 달성할 때까지 계속 수행해야 함
        # -> 차라리 average IID distance가 특정 값 아래로 내려갈 때까지 달성하는게 낫지 않나?
        logging.info("Diffusion efficiency: {}".format(DE))
        if DE <= self.params.epsilon:
            # break         # TODO: 반복문으로 할거면 break
            return None, None  # TODO: FedDif diffusion 별로 수행할거면 여기서 return

        # [Step 2-4] Model transmission
        ## TODO: 수형 요구사항
        ## policy 대로 transmission을 수행할 때 다음의 것들을 해주셔야 합니다.
        ## 1. self.visit(model.id, user.id)
        ##   -> 각 model별로 방문한 user들에 대한 subchain 방문 여부를 업데이트 해야 합니다. (value, weight 계산에 쓰임)
        for i in range(self.n_model):
            if policy_model[i] != -1:
                self.visit(i, policy_model[i])
        ## 2. 각 user별 model 세팅
        ##   -> 이 때 tranmission 하고 받지 않은 놈은 local_model이 None이어야 합니다. : TODO: 왜 user별 model이 세팅되어야 하는거지?
        # for i in range(self.n_model):
        #     if policy_model[i] == -1:
        #         self.PUE_list[policy_model[i]]
        ## 3. 각 model별 업데이트
        ##   -> model 별로 prev_DoL, subchain, DSI 반영한 DoL 업데이트
        ##   -> model 별로 curr_trainer, prev_trainer 업데이트
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

        return IID_dist, DE

    def random_diffusion(self):
        random.shuffle(self.idx)
        for i in range(self.n_model):
            self.models[i].prev_trainer = self.models[i].curr_trainer
            self.models[i].curr_trainer = self.idx[i]
            self.models[i].diffusion_subchain.append(self.idx[i])

    def local_training(self, lr):
        for i in range(self.n_model):
            if self.curr_policy[i] is True:
                model = self.models[i]
                model.optimizer.param_groups[0]['lr'] = lr
                model.local_train(self.PUE_list[model.curr_trainer].trainloader, self.model_dir)

    def global_aggregation(self):
        data_size = self.get_data_size()
        aggregated_dict = FedAvg(self.global_model, self.models, data_size, self.params)
        self.global_model.load_state_dict(aggregated_dict)
        self.save_checkpoint(self.save_dir)

    def evaluate(self):
        logging.info("Global model evaluation")
        global_valid_metrics = evaluate(self.global_model, utils.loss_function, self.testloader, utils.metrics, self.params)
        global_acc_temp = global_valid_metrics['accuracy']
        global_loss_temp = global_valid_metrics['loss']
        return global_acc_temp, global_loss_temp
        # self.write_csv(global_name, global_data, global_fieldnames)

    def save_checkpoint(self, save_dir):
        ### Save the parameters of the global model
        torch.save(self.global_model.state_dict(), os.path.join(save_dir, "global_model.pth"))
        self.s_global_model = os.path.getsize(os.path.join(save_dir, "global_model.pth"))

    def set_testloader(self, testloader):
        self.testloader = testloader