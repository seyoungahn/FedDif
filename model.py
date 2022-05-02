import copy
import logging
import methods
import numpy as np
import utils
import torch.optim as optim

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

    def clear_model(self):
        self.diffusion_round = 0
        self.prev_DoL = None
        self.curr_DoL = None
        self.prev_trainer = None
        self.curr_trainer = None
        self.diffusion_subchain = []
        self.subchain_datasize = 0

    def get_next_DoL(self, DSI, datasize):
        return (self.subchain_datasize * self.curr_DoL + datasize * DSI) / (self.subchain_datasize + datasize)

    def local_configuration(self):
        self.optimizer = optim.SGD(self.ML_model.parameters(), lr=self.params.t_learning_rate, momentum=self.params.t_momentum, weight_decay=self.params.t_weight_decay)
        if self.params.t_lr_scheduler == 1:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda round: self.params.t_learning_rate / (1 + round))

    def local_train(self, trainloader, model_dir):
        logging.info("\t+ {}th local model training...".format(self.id+1))
        logging.info("\t+ Diffusion chain: {}".format(self.diffusion_subchain))
        for epoch in range(self.params.t_local_epochs):
            logging.info("\t  => Epoch {}/{}".format(epoch, self.params.t_local_epochs))
            train_metrics = methods.FedAvg.train(self.ML_model, self.optimizer, self.lr_scheduler, utils.loss_function, trainloader, utils.metrics, self.params)

        utils.save_local_checkpoint({
            'state_dict': self.ML_model.state_dict()
        }, checkpoint=model_dir, party_idx=self.id)
        return train_metrics