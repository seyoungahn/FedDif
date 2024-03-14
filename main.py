from experiment import *
import torch
import sys

# Experiments
if __name__ == "__main__":
    # Hyperparameter setting
    json_path = os.path.join("params.json")
    assert os.path.isfile(json_path), "No JSON configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    args = sys.argv
    params.t_gpu_no = int(args[1])
    params.alpha = float(args[2])
    params.epsilon = float(args[3])
    params.t_local_epochs = int(args[4])
    params.scenario = int(args[5]) # 0: CIFAR100 / 1: EMNIST / 2: CIFAR10

    if params.scenario == 0:
        ####### CIFAR-100
        params.t_model_version = 'lenet'
        params.t_dataset_type = 'cifar100'
        params.t_class_num = 100
    elif params.scenario == 1:
        ####### EMNIST
        params.t_model_version = 'fcn'
        params.t_dataset_type = 'emnist'
        params.t_class_num = 47
    elif params.scenario == 2:
        ####### CIFAR-10
        params.t_model_version = 'lenet'
        params.t_dataset_type = 'cifar10'
        params.t_class_num = 10
    elif params.scenario == 3:
        ####### Tiny ImageNet
        params.t_model_version = 'resnet18'
        params.t_dataset_type = 'tiny-imagenet'
        params.t_class_num = 200
    else:
        assert "Settings error"

    experiment_name = 'FedDif_cuda-{}_alpha-{}_eps-{}_locepoch-{}_PUE-{}%_{}_{}'.format(params.t_gpu_no, params.alpha, params.epsilon, params.t_local_epochs, params.r_model * 100, params.t_dataset_type, params.t_model_version)
    print("Experiment name: " + experiment_name)
    e1 = Experiment(experiment_name, params)
    e1.FedDif()
    del e1

