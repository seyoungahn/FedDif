import json
import logging
import os
import shutil
import torch
import errno
import csv
import math

# import tensorflow as tf
import numpy as np
import scipy.misc
import scipy.stats

import torch.nn as nn
import torch.nn.functional as F

try:
    from StringIO import StringIO   # Python 2.7
except ImportError:
    from io import BytesIO          # Python 3.x

class Params():
    """
    Class that loads hyperparameters from a json file.
    Example:
        params = Params(json_path)
        print(params.learning_rate)
        params.learning_rate = 0.5  # change the value of learning_rate in params
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        ## Loads parameters from json file
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        ## Gives dict-like access to Params instance by 'params.dict['learning_rate']
        return self.__dict__

class RunningAverage():
    """
    A simple class that maintains the running average of a quantity
    Example:
        avg_loss = RunningAverage()
        avg_loss.update(2)
        avg_loss.update(4)
        avg_loss() = 3
    """
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total/float(self.steps)

def set_logger(log_path):
    """
    Set the logger to log info in terminal and file 'log_path'.

    In general, it is useful to have a logger so that every output to the terminal is saved in a permanent file.
    Here we save it to 'model_dir/train.log'.
    Example:
        logging.info("Start training...")
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

def save_dict_to_json(d, json_path):
    """
    Saves dict of floats in json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)

def save_local_checkpoint(state, checkpoint, party_idx):
    """
    Saves model and training parameters at checkpoint + 'last.pth.tar'.
    If is_best==True, also saves checkpoint + 'best.pth.tar'
    :param state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
    :param is_best: (bool) True if it is the best model seen till now
    :param checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'party_' + str(party_idx) + '_last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint directory does not exist: making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint directory exists.")
    torch.save(state, filepath)
    # if is_best:
    #     shutil.copyfile(filepath, os.path.join(checkpoint, 'party_' + str(party_idx) + '_best.pth.tar'))

def save_checkpoint(state, is_best, checkpoint):
    """
    Saves model and training parameters at checkpoint + 'last.pth.tar'.
    If is_best==True, also saves checkpoint + 'best.pth.tar'
    :param state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
    :param is_best: (bool) True if it is the best model seen till now
    :param checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint directory does not exist: making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint directory exists.")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))

def load_checkpoint(checkpoint, model, optimizer=None):
    """
    Loads model parameters (state_dict) from file_path.
    If optimizer is provided, loads state_dict of optimizer assuming it is present in checkpoint.
    :param checkpoint: (string) filename which needs to be loaded
    :param model: (torch.nn.Module) model for which the parameters are loaded
    :param optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise("Checkpoint file does not exist {}".format(checkpoint))
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint)
    else:
        # This helps avoid errors when loading single-GPU-trained weights onto CPU-model
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint

def loss_function(outputs, labels):
    """
    Compute the cross-entropy loss given outputs and labels
    :param outputs: (Variable) dimension batch_size x 6 - output of the model
    :param labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]
    :return: loss (Variable) cross-entropy loss for all images in the batch
    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions.
          This example demonstrates how you can easily define a custom loss function
    """
    return nn.CrossEntropyLoss()(outputs, labels)

def loss_function_kd(outputs, labels, teacher_outputs, params):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    Hyperparameters: temperature, alpha
    Note: the KL Divergence for PyTorch comparing the softmaxs of teacher and student expects the input tensor to be log probabilities
    """
    alpha = params.alpha
    T = params.temperature
    # KLDivergence issue: reduction='mean' doesn't return the true KL divergence value
    #                     please use reduction = 'batchmean' which aligns with KL math definition.
    #                     In the next major release, 'mean' will be changed to be the same as 'batchmean'
    KD_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs/T, dim=1), F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + F.cross_entropy(outputs, labels) * (1. - alpha)
    return KD_loss

def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.
    :param outputs: (np.ndarray) output of the model
    :param labels: (np.ndarray) [0, 1, ..., num_classes-1]
    :return: (float) accuracy in [0, 1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs==labels) / float(labels.size)

def mkdir_p(path):
    ''' make dir if not exist '''
    try:
        os.makedirs(path)
    except OSError as exc: # Python > 2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def write_csv(save_dir, file_name, data, fieldnames):
    with open(save_dir + "/" + file_name + ".csv", 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        data_dict = {}
        l_data = len(data)
        for i in range(l_data):
            _field = fieldnames[i]
            _data = data[i]
            data_dict[_field] = _data
        writer.writerow(data_dict)

# maintain all metrics required in this dictionary - these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}

def DoL_operation(p, datasize_p, q, datasize_q):
    return (datasize_p * p + datasize_q * q) / (datasize_p + datasize_q)

def prob_dist(p, q, type='l2'):
    if type == 'l2':
        return np.linalg.norm(p - q, ord=2)
    elif type == 'KLD':
        return np.sum(np.where(p != 0.0, p * np.log(p / q), 0.0))
    elif type == 'JSD':
        return (1 / 2) * KL_divergence(p, (p + q) / 2) + (1 / 2) * KL_divergence(q, (p + q) / 2)
    elif type == 'EMD':
        return scipy.stats.wasserstein_distance(p, q)

def KL_divergence(p, q):
    return np.sum(np.where(p != 0.0, p * np.log(p / q), 0.0))

def RMSE(p, q):
    return np.sqrt(np.mean((p - q)**2))

def EMD(p, q):
    return scipy.stats.wasserstein_distance(p, q)

def JSD(p, q):
    return (1/2) * KL_divergence(p, (p+q)/2) + (1/2) * KL_divergence(q, (p+q)/2)

def Euclidean(x1, y1, x2, y2):
    return (x1 - x2) ** 2 + (y1 - y2) ** 2

def dist_node(node1, node2):
    return ((node1.x - node2.x)**2 + (node1.y - node2.y)**2 + 23.5**2)**0.5

def SNR_user(params, tx, rx, num_RBs, interference=False):
    pathloss_db = params.s_beta_zero - 10.0 * params.s_kappa * math.log10(dist_node(tx, rx))
    SNR_dB = tx.tx_power + pathloss_db - params.s_noise
    return SNR_dB

def SINR_user(params, tx, rx, inter_tx, num_RBs):
    pathloss = params.s_beta_zero - 10.0 * params.s_kappa * math.log10(dist_node(tx, rx))
    noise = params.s_noise_power
    if num_RBs > 0:
        noise = params.s_noise_power + 10.0 * math.log10(params.s_subcarrier_bandwidth * params.s_n_subcarrier_RB * num_RBs)
    inter_pathloss = params.s_beta_zero - 10.0 * params.s_kappa * math.log10(dist_node(inter_tx, rx))
    INR_dB = 10.0 * math.log10(10.0 ** ((inter_tx.tx_power - inter_pathloss) / 10.0) + 10.0 ** (noise / 10.0))
    SINR_dB = tx.tx_power + pathloss - INR_dB
    return SINR_dB

def spectral_efficiency_user(params, tx, rx, num_RBs, interference=False, inter_tx=None):
    SINR_dB = SNR_user(params, tx, rx, num_RBs)
    if interference is True:
        SINR_dB = SINR_user(params, tx, rx, inter_tx, num_RBs)
    SINR = 10.0**(SINR_dB / 10.0)
    spectral_efficiency = math.log2(1.0 + SINR)
    return spectral_efficiency

def datarate(params, tx, rx, num_RBs, interference=False, inter_tx=None):
    result = num_RBs * params.s_subcarrier_bandwidth * params.s_n_subcarrier_RB * spectral_efficiency_user(params, tx, rx, num_RBs, interference=interference, inter_tx=inter_tx)
    return result
