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

    # Epsilon test
    params.t_gpu_no = int(args[1])
    task_no = int(args[2])

    if task_no == 1:
        ## Exp. 1: Concentration parameter test
        for alpha in [0.1]:
            params.alpha = alpha
            params.bar_delta = 0.2
            params.s_QoS_th = 1.0
            experiment_name = 'FedDif_v4_cuda-{}_alpha-{}_cifar10_cnn'.format(int(args[1]), alpha)
            print("Experiment name: " + experiment_name)
            e1 = Experiment(experiment_name, params)
            e1.FedDif()
            del e1

            params.epsilon = 1.0
            experiment_name = 'FedDif_v4_FedAvg_cuda-{}_alpha-{}_cifar10_cnn'.format(int(args[1]), alpha)
            print("Experiment name: " + experiment_name)
            e2 = Experiment(experiment_name, params)
            e2.FedDif()
            del e2
    elif task_no == 2:
        ## Exp. 2: Minimum tolerable diffusion efficiency test
        for bar_delta in [0.04]:
            params.alpha = 1.0
            params.bar_delta = bar_delta
            params.s_QoS_th = 0.0
            experiment_name = 'FedDif_v6_cuda-{}_delta-{}_cifar10_cnn'.format(int(args[1]), bar_delta)
            print("Experiment name: " + experiment_name)
            e = Experiment(experiment_name, params)
            e.FedDif()
            del e
        # ## Exp. 2: Weight initialization methods test
        # for weight_init in ["xavier-normal", "xavier-uniform", "he-normal", 'he-uniform']:
        #     params.alpha = 1.0
        #     params.epsilon = 0.0
        #     params.s_QoS_th = 1.0
        #     params.t_init = weight_init
        #     experiment_name = 'FedDif_v4_cuda-{}_{}_cifar10_cnn'.format(int(args[1]), weight_init)
        #     print("Experiment name: " + experiment_name)
        #     e = Experiment(experiment_name, params)
        #     e.FedDif()
        #     del e
    elif task_no == 3:
        ## Exp. 3: Minimum tolerable QoS test
        for s_QoS_th in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]:
            params.alpha = 1.0
            params.epsilon = 0.0
            params.s_QoS_th = s_QoS_th
            experiment_name = 'FedDif_v4_cuda-{}_QoS-{}_cifar10_cnn'.format(int(args[1]), s_QoS_th)
            print("Experiment name: " + experiment_name)
            e = Experiment(experiment_name, params)
            e.FedDif()
            del e
    elif task_no == 4:
        ## Exp. 4: ML task test
        # [('cnn', 'cifar10'), ('lstm', 'fmnist'), ('fcn', 'mnist')]
        for t_model_version, t_dataset_type in [('resnet34', 'cifar10')]:
            params.t_model_version = t_model_version
            params.t_dataset_type = t_dataset_type
            params.alpha = 1.0
            params.epsilon = 0.0
            params.s_QoS_th = 1.0
            experiment_name = 'FedDif_v4_FedDif_cuda-{}_{}_{}'.format(int(args[1]), t_dataset_type, t_model_version)
            print("Experiment name: " + experiment_name)
            e1 = Experiment(experiment_name, params)
            e1.FedDif()
            del e1

            params.epsilon = 1.0
            experiment_name = 'FedDif_v4_FedAvg_cuda-{}_{}_{}'.format(int(args[1]), t_dataset_type, t_model_version)
            print("Experiment name: " + experiment_name)
            e2 = Experiment(experiment_name, params)
            e2.FedDif()
            del e2
    elif task_no == 5:
        ## Exp. 5: Fully-decentralized FL test
        print("To be discussed..")
        pass
    elif task_no == 6:
        ## Emergence test
        params.alpha = 1.0
        params.epsilon = 0.0
        params.s_QoS_th = 1.0
        experiment_name = 'FedDif_test_radar1d_conv1d'
        print("Experiment name: " + experiment_name)
        e = Experiment(experiment_name, params)
        e.FedDif()
        del e


    # for epsilon in [0.002, 0.003, 0.004, 0.005, 0.006, 0.01, 0.02, 0.03]:
    #     params.epsilon = epsilon
    #     experiment_name = 'FedDif_cuda-{}_epsilon-{}_cifar10_cnn'.format(int(args[1]), epsilon)
    #     print("Experiment name: " + experiment_name)
    #     e = Experiment(experiment_name, params)
    #     e.FedDif()
    #     del e

    # params.t_dataset_type = "mnist"
    # params.t_model_version = "fcn"
    #
    # experiment_name = 'FedDif-test_mnist_fcn'
    # print("Experiment name is \"" + experiment_name + "\"")
    # e = Experiment(experiment_name, params)
    # e.FedDif()
    # del e

    # experiment_name = 'FedDif_init-once_cifar10_cnn'
    # print("Experiment name is \"" + experiment_name + "\"")
    # e = Experiment(experiment_name, params)
    # e.FedDif()
    # del e

    # Epsilon test
    # for epsilon in [0.002, 0.004, 0.006, 0.008, 0.01, 0.02, 0.03]:
    #     params.epsilon = epsilon
    #     experiment_name = 'FedDif_epsilon-{}_cifar10_cnn'.format(epsilon)
    #     print("Experiment name: " + experiment_name)
    #     e = Experiment(experiment_name, params)
    #     e.FedDif()
    #     del e

    ## Weight initialization test
    # for init in ['xavier-uniform', 'xavier-normal', 'he-uniform', 'he-normal']:
    #     params.t_init = init
    #     experiment_name = 'FedDif_{}_cifar10_cnn'.format(init)
    #     print("Experiment name: " + experiment_name)
    #     e = Experiment(experiment_name, params)
    #     e.FedDif()
    #     del e

    ## Weight initialization test
    # for alpha in [0.9, 0.7, 0.5, 0.3, 0.1]:
    #     params.alpha = alpha
    #     experiment_name = 'FedDif_init-once_alpha-{}_cifar10_cnn'.format(alpha)
    #     print("Experiment name: " + experiment_name)
    #     e = Experiment(experiment_name, params)
    #     e.FedDif()
    #     del e

    ## Weight difference test
    # for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
    #     params.alpha = alpha
    #     experiment_name = 'FedDif_weight_difference_alpha-{}_cifar10_cnn'.format(alpha)
    #     print("Experiment name: " + experiment_name)
    #     e = Experiment(experiment_name, params)
    #     e.FedDif_weight_difference()
    #     del e

    ## Probability distance test
    # params.t_dist = dist_type
    # params.t_gpu_no = gpu
    # if dist_type == 'KLD':
    #     epsilons = [1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5, 10e-5, 55e-5]
    # elif dist_type == 'JSD':
    #     epsilons = [1e-6, 2e-6, 3e-6, 4e-6, 5e-6, 6e-6, 7e-6, 8e-6, 9e-6, 55e-5, 65e-5]
    # elif dist_type == 'EMD':
    #     epsilons = [1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5, 10e-5, 9e-6]
    #
    # for epsilon in epsilons:
    #     params.epsilon = epsilon
    #     experiment_name = 'FedDif_prob-dist-{}_epsilon-{}_cifar10_cnn'.format(dist_type, epsilon)
    #     print("Experiment name: " + experiment_name)
    #     e = Experiment(experiment_name, params)
    #     e.FedDif()
    #     del e
