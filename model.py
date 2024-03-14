import copy
import logging
import methods
import numpy as np
import utils
import torch.optim as optim
import torch

class Model:
    def __init__(self, id, params):
        self.id = id
        self.params = params
        self.ML_model = None
        self.diffusion_round = 0
        self.optimizer = None
        self.lr_scheduler = None
        self.prev_DoL = None
        self.curr_DoL = None
        self.prev_trainer = None
        self.curr_trainer = None
        self.diffusion_subchain = []
        self.subchain_datasize = 0
        self.curr_state_gradient_diff = None
        self.prev_state_gradient_diff = None
        self.parameter_drifts = None
        self.global_model = None

    def clear_config(self):
        self.diffusion_round = 0
        self.prev_DoL = None
        self.curr_DoL = None
        self.prev_trainer = None
        self.curr_trainer = None
        self.diffusion_subchain = []
        self.subchain_datasize = 0

    def get_next_DoL(self, DSI, datasize):
        return (self.subchain_datasize * self.curr_DoL + datasize * DSI) / (self.subchain_datasize + datasize)

    def local_train(self, trainloader, data_diff, lr, local_update_last, global_update_last, parameter_drift, diff_rnd):
        logging.info("\t+ {}th local model training...".format(self.id+1))
        logging.info("\t+ Diffusion chain: {}".format(self.diffusion_subchain))

        prev_model_param = utils.get_mdl_params([self.ML_model])[0]
        prev_mdl = torch.tensor(prev_model_param, dtype=torch.float32).to(self.params.t_gpu_no)
        # prev_mdl = utils.get_mdl_tensors(self.ML_model).to(self.params.t_gpu_no)
        hist_i = torch.tensor(parameter_drift, dtype=torch.float32).to(self.params.t_gpu_no)

        self.ML_model, train_metrics, loss_record = methods.FedAvg.train_FedDif(diff_rnd, self.ML_model, trainloader, lr, self.params, data_diff, local_update_last, global_update_last, hist_i, prev_mdl)

        curr_model_param = utils.get_mdl_params([self.ML_model])[0]
        delta_param_curr = curr_model_param - prev_model_param

        return train_metrics, loss_record, delta_param_curr

    def evaluate(self, testloader):
        local_valid_metrics = methods.FedAvg.evaluate(self.ML_model, utils.loss_function, testloader, utils.metrics, self.params)
        logging.info("\t  => Test acc: {:.4f} / Test loss: {:.4f}".format(local_valid_metrics['accuracy'], local_valid_metrics['loss']))