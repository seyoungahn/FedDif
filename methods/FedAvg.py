import copy

import utils
from tqdm import tqdm
import torch
import numpy as np
import logging

max_norm = 10

# def FedAvg(global_model, models, data_sizes, param_drifts):
#     # Objective: aggregates all local model to the global model
#     # Inputs: global model, a list of secondary UEs, experiment parameters
#     # Outputs: parameter dictionary of aggregated model
#     global_model_dict = dict(global_model.state_dict())
#     aggregated_dict = dict(global_model.state_dict())
#     parameter_drifts_dict = dict(param_drifts.state_dict())
#     parties_dict = {}
#
#     for i, model in enumerate(models):
#         parties_dict[i] = dict(model.ML_model.state_dict())
#     data_ratio = data_sizes / sum(data_sizes)
#
#     print("\t+ Data_sizes: ", data_sizes)
#     print("\t+ Sum: {}".format(sum(data_sizes)))
#     print("\t+ Data ratio: ", data_ratio)
#
#     for name, param in global_model_dict.items():
#         aggregated_dict[name].data.copy_(sum([data_ratio[i] * parties_dict[i][name].data for i in range(len(models))]) + parameter_drifts_dict[name].data)
#     return aggregated_dict

def FedAvg(global_model, models, data_sizes, param_drifts):
    # Objective: aggregates all local model to the global model
    # Inputs: global model, a list of secondary UEs, experiment parameters
    # Outputs: parameter dictionary of aggregated model
    global_model_dict = dict(global_model.state_dict())
    aggregated_dict = dict(global_model.state_dict())
    parameter_drifts_dict = dict(param_drifts.state_dict())
    parties_dict = {}

    for i, model in enumerate(models):
        parties_dict[i] = dict(model.ML_model.state_dict())
    data_ratio = data_sizes / sum(data_sizes)
    print("\t+ Data_sizes: ", data_sizes)
    print("\t+ Sum: {}".format(sum(data_sizes)))
    print("\t+ Data ratio: ", data_ratio)
    for name, param in global_model_dict.items():
        aggregated_dict[name].data.copy_(sum([data_ratio[i] * parties_dict[i][name].data for i in range(len(models))]) + parameter_drifts_dict[name].data)
    return aggregated_dict

def train(model, optimizer, scheduler, criterion, trainloader, metrics, params):
    model.train()

    summaries = []
    avg_loss = utils.RunningAverage()

    # Use tqdm for progress bar
    with tqdm(total=len(trainloader) * params.t_local_epochs) as t:
        for loc_epoch in range(params.t_local_epochs):
            for batch_idx, (x_train, y_train) in enumerate(trainloader):
                # Move to GPU if available
                if params.t_cuda:
                    x_train, y_train = x_train.cuda(non_blocking=True), y_train.cuda(non_blocking=True)

                x_train, y_train = torch.autograd.Variable(x_train), torch.autograd.Variable(y_train)

                y_pred = model(x_train)
                loss = criterion(y_pred, y_train)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if params.t_lr_scheduler == 1:
                    scheduler.step()

                # Evaluation (train acc, train loss)
                if batch_idx % params.t_save_summary_steps == 0:
                    y_pred = y_pred.data.cpu().numpy()
                    y_train = y_train.data.cpu().numpy()

                    summary = {metric: metrics[metric](y_pred, y_train) for metric in metrics}
                    summary['loss'] = loss.item()
                    summaries.append(summary)

                avg_loss.update(loss.item())

                t.set_postfix(loss='{:05.3f}'.format(avg_loss()))
                t.update()
            # print(summaries)
            metrics_mean = {metric:np.mean([x[metric] for x in summaries]) for metric in summaries[0]}
        return metrics_mean

def train_FedDif(diff_rnd, model, trainloader, lr, params, data_dist, local_update_last, global_update_last, hist_i, global_model_param):
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=params.t_weight_decay)
    # upper_bound_grad = 1
    # lipschitz_const = 0.5
    IID = np.array([1.0 / params.t_class_num for _ in range(params.t_class_num)])

    state_update_diff = torch.tensor(global_update_last - local_update_last, dtype=torch.float32).to(params.t_gpu_no)
    # print(np.sum(global_update_last - local_update_last, axis=0))

    avg_loss_f_i_val = 0.0
    avg_loss_val = 0.0
    avg_reg_val = 0.0
    avg_grad_corr_val = 0.0
    avg_IID_reg_val = 0.0
    cnt = 0
    n_samples = len(trainloader.dataset)
    n_pars = utils.get_mdl_params([model]).shape[1]

    if params.scenario == 0: ## LeNet + CIFAR100
        a = 0.01
        b = 0.1
    elif params.scenario == 1: ## FCN + EMNIST
        a = 0.02 # 0.02
        b = 0.1 # 0.1
    elif params.scenario == 2: ## LeNet + CIFAR10
        a = 0.005 # 0.1
        b = 0.1 # 0.01
    else:
        assert "SCENARIO ERROR"

    # for model_params in model.parameters():
    #     model_params.required_grad = True

    model.train()
    for loc_epoch in range(params.t_local_epochs):
        epoch_loss = 0
        train_iter = trainloader.__iter__()
        for i in range(len(trainloader)):
            x_train, y_train = train_iter.__next__()
            x_train = x_train.to(params.t_gpu_no)
            y_train = y_train.to(params.t_gpu_no)

            y_pred = model(x_train)

            loss_f_i = criterion(y_pred, y_train.reshape(-1).long())

            loss_f_i = loss_f_i / list(y_train.size())[0]

            # local_parameter = torch.tensor(utils.get_mdl_params([model])[0], dtype=torch.float32).to(params.t_gpu_no)

            local_parameter = None
            for param in model.parameters():
                if not isinstance(local_parameter, torch.Tensor):
                    local_parameter = param.reshape(-1)
                else:
                    local_parameter = torch.cat((local_parameter, param.reshape(-1)), 0)

            # local_parameter = utils.get_mdl_tensors(model).to(params.t_gpu_no)

            # local_parameter = utils.get_mdl_params([model])[0]
            # local_parameter = torch.tensor(local_parameter, dtype=torch.float32).to(params.t_gpu_no)

            # print(state_update_diff)
            # regularization_term = torch.linalg.vector_norm(local_update_last - local_parameter)
            # regularization_term = (alpha / 2) * torch.tensor(params.t_class_num * utils.prob_dist(data_dist, IID, params.t_dist), dtype=torch.float32).to(params.t_gpu_no)
            # regularization_term = torch.tensor(utils.prob_dist(DoL, IID, params.t_dist), dtype=torch.float32).to(params.t_gpu_no)

            # gradient_correction_term = torch.sum(local_parameter * state_gradient_diffs)

            # regularization_term = alpha / 2 * torch.sum((local_parameter - (glo)))
            # regularization_term = alpha / 2 * torch.tensor(params.t_class_num * utils.prob_dist(data_dist, IID, params.t_dist), dtype=torch.float32).to(params.t_gpu_no)
            # regularization_term = a / 2 * torch.sum((hist_i - state_update_diff) * (hist_i - state_update_diff)) + b / 2 * torch.tensor(params.t_class_num * utils.prob_dist(data_dist, IID, params.t_dist), dtype=torch.float32).to(params.t_gpu_no)

            # regularization_term = a / 2 * torch.sum((local_parameter - (global_update_last - hist_i)) * (local_parameter - (global_update_last - hist_i)))
            # regularization_term = a / 2 * torch.sum(local_parameter * state_update_diff) * torch.tensor(params.t_class_num * utils.prob_dist(data_dist, IID, params.t_dist), dtype=torch.float32).to(params.t_gpu_no)

            '''
            # print("IID_dist: {}".format(IID_dist))
            # print("Local drifts: {}".format(local_drifts))

            regularization_term = a * (local_drifts - IID_dist)
            '''
            # if diff_rnd == 0:
            #     IID_reg_term = torch.tensor(1.0, dtype=torch.float32).to(params.t_gpu_no)
            # else:
            #     IID_reg_term = (b / 2) * torch.tensor(params.t_class_num * utils.prob_dist(data_dist, IID, params.t_dist), dtype=torch.float32).to(params.t_gpu_no)

            IID_reg_term = b / 2 * torch.tensor(params.t_class_num * utils.prob_dist(data_dist, IID, params.t_dist), dtype=torch.float32).to(params.t_gpu_no)

            # temp = torch.sum(local_parameter - global_model_param)
            # print(temp.item())

            # regularization_term = a / 2 * torch.sum((hist_i + local_parameter - global_model_param) * (hist_i + local_parameter - global_model_param))
            # regularization_term = (a / 2) * IID_reg_term * torch.sum((hist_i + local_parameter - global_model_param) * (hist_i + local_parameter - global_model_param))
            regularization_term = a / (2 * IID_reg_term) * torch.sum((hist_i + local_parameter - global_model_param) * (hist_i + local_parameter - global_model_param))

            gradient_correction_term = torch.sum(local_parameter * state_update_diff)

            # loss = loss_f_i + regularization_term + gradient_correction_term
            # loss = loss_f_i + regularization_term + gradient_correction_term + IID_reg_term
            loss = loss_f_i + regularization_term + gradient_correction_term
            # loss = loss_f_i + regularization_term
            # loss = loss_f_i + gradient_correction_term
            # loss = loss_f_i

            avg_loss_f_i_val += loss_f_i.item()
            avg_loss_val += loss.item()
            avg_reg_val += regularization_term.item()
            avg_grad_corr_val += gradient_correction_term.item()
            avg_IID_reg_val += IID_reg_term.item()
            cnt += 1

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm)
            optimizer.step()
            epoch_loss += loss.item() * list(y_train.size())[0]

        if (loc_epoch+1) % params.t_print_step == 0:
            epoch_loss /= n_samples
            # if params.t_weight_decay != None:
                # Add L2 loss to complete f_i
                # model_params = utils.get_mdl_params([model], n_pars)
                # epoch_loss += params.t_weight_decay / 2 * np.sum(model_params * model_params)
            # loss_train, acc_train = utils.get_acc_loss(trainloader, model, params)
            print("\t\tEPOCH %02d, Training Loss: %.4f, LR: %.5f" % (loc_epoch+1, epoch_loss, lr))
            model.train()

    # Freeze model
    for model_params in model.parameters():
        model_params.required_grad = False
    model.eval()

    avg_loss_f_i_val /= cnt
    avg_loss_val /= cnt
    avg_reg_val /= cnt
    avg_grad_corr_val /= cnt
    avg_IID_reg_val /= cnt
    loss_record = [avg_loss_f_i_val, avg_reg_val, avg_grad_corr_val, avg_loss_val, avg_IID_reg_val]
    print("%.04f | (%.04f / %.04f) | %.04f || %.04f" % (avg_loss_f_i_val, avg_reg_val, avg_grad_corr_val, avg_IID_reg_val, avg_loss_val))
    metrics_mean = 0
    return model, metrics_mean, loss_record

def evaluate(model, criterion, validloader, metrics, params):
    """
    Evaluate the model on 'num_steps' batches
    :param model: (torch.nn.Module) the neural network
    :param criterion: a function that takes y_pred and y_valid and computes the loss for the batch
    :param validloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
    :param metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
    :param params: (Params) hyperparameters
    :return:
    """
    # Set model to evaluation mode
    model.eval()

    # Summary for current eval loop
    summaries = []

    # Compute metrics over the dataset
    for batch_idx, (x_valid, y_valid) in enumerate(validloader):
        # Move to GPU if available
        if params.t_cuda:
            x_valid, y_valid = x_valid.to(params.t_gpu_no), y_valid.to(params.t_gpu_no)
        # Fetch the next evaluation batch
        x_valid, y_valid = torch.autograd.Variable(x_valid), torch.autograd.Variable(y_valid)

        # Compute model output
        y_pred = model(x_valid)
        loss = criterion(y_pred, y_valid)

        # Extract data from torch Variable, move to CPU, convert to numpy arrays
        y_pred = y_pred.data.cpu().numpy()
        y_valid = y_valid.data.cpu().numpy()

        # Compute all metrics on this batch
        summary = {metric: metrics[metric](y_pred, y_valid) for metric in metrics}
        summary['loss'] = loss.item()
        summaries.append(summary)

    # Compute mean of all metrics in summary
    # print("Evaluation summaries")
    # print(summaries)
    metrics_mean = {metric: np.mean([x[metric] for x in summaries]) for metric in summaries[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    # logging.info("- Eval metrics: " + metrics_string)
    return metrics_mean

def weight_difference(target_model, base_model):
    wd = 0
    cnt = 0
    for target_param, base_param in zip(target_model.parameters(), base_model.parameters()):
        wd += torch.norm(target_param.data - base_param.data)
        cnt += 1
    # print("Weight difference: {}|cnt: {}".format(wd, cnt))
    return wd

def weight_error(target_model, base_model):
    we = 0
    cnt = 0
    for target_param, base_param in zip(target_model.parameters(), base_model.parameters()):
        we += torch.norm(target_param.data - base_param.data) / torch.norm(base_param.data)
        cnt += 1
    # print("Weight error: {}|cnt: {}".format(we, cnt))
    return we