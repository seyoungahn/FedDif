import utils
from tqdm import tqdm
import torch
import numpy as np
import logging

def FedAvg(global_model, models, data_sizes, params):
    # Objective: aggregates all local model to the global model
    # Inputs: global model, a list of secondary UEs, experiment parameters
    # Outputs: parameter dictionary of aggregated model
    global_model_dict = dict(global_model.state_dict())
    aggregated_dict = dict(global_model.state_dict())
    parties_dict = {}
    for i in range(len(models)):
        parties_dict[i] = dict(models[i].ML_model.state_dict())
    beta = data_sizes / sum(data_sizes)
    print("Data_sizes: ", data_sizes)
    print("Sum: {}".format(sum(data_sizes)))
    print("Beta: ", beta)
    for name, param in global_model_dict.items():
        aggregated_dict[name].data.copy_(sum([beta[i] * parties_dict[i][name].data for i in range(len(models))]))

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

def train_FedDif(model, optimizer, criterion, trainloader, metrics, params):
    # model.train()

    summaries = []
    avg_loss = utils.RunningAverage()

    # Use tqdm for progress bar
    with tqdm(total=len(trainloader) * params.t_local_epochs) as t:
        for loc_epoch in range(params.t_local_epochs):
            for batch_idx, (x_train, y_train) in enumerate(trainloader):
                # Move to GPU if available
                if params.t_cuda:
                    x_train, y_train = x_train.to(params.t_gpu_no), y_train.to(params.t_gpu_no)

                x_train, y_train = torch.autograd.Variable(x_train), torch.autograd.Variable(y_train)

                y_pred = model(x_train)
                loss = criterion(y_pred, y_train)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

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
            metrics_mean = {metric: np.mean([x[metric] for x in summaries]) for metric in summaries[0]}
        return metrics_mean

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
    logging.info("- Eval metrics: " + metrics_string)
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