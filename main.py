from experiment import *
import torch

# Experiments
if __name__ == "__main__":
    # Hyperparameter setting
    json_path = os.path.join("params.json")
    assert os.path.isfile(json_path), "No JSON configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # params.t_dataset_type = "mnist"
    # params.t_model_version = "fcn"
    #
    # experiment_name = 'FedDif-test_mnist_fcn'
    # print("Experiment name is \"" + experiment_name + "\"")
    # e = Experiment(experiment_name, params)
    # e.FedDif()
    # del e

    params.t_dataset_type = "fmnist"
    params.t_model_version = "lstm"

    experiment_name = 'FedDif-test_fmnist_lstm'
    print("Experiment name is \"" + experiment_name + "\"")
    e = Experiment(experiment_name, params)
    e.FedDif()
    del e
