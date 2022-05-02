"""
CIFAR-10 data normalization reference:
https://github.com/Armour/pytorch-nn-practice/blob/master/utils/meanstd.py
"""

import random
import os
import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import logging
from tqdm import trange

class DatasetSplit(torch.utils.data.Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        data, label = self.dataset[self.idxs[item]]
        return torch.as_tensor(data), torch.as_tensor(label)

def getDirichletData(data, prior_dist, alpha, n_classes):
    n_users = len(prior_dist)
    label_list = np.array(data.targets)
    min_size = 0
    np.random.seed(2020)

    net_dataidx_map = {}
    while min_size < n_classes:
        idx_batch = [[] for _ in range(n_users)]
        # for each class in the dataset
        for k in range(n_classes):
            idx_k = np.where(label_list == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, n_users))
            ## Balance
            proportions = np.array([p * (len(idx_j) < len(label_list) / n_users) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(n_users):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]

    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(label_list[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    # print('Data statistics: %s' % str(net_cls_counts))

    DSI_list = [[0.0 for _ in range(n_classes)] for _ in range(n_users)]

    for i in range(n_users):
        for j in net_cls_counts[i]:
            DSI_list[i][j] += net_cls_counts[i][j]
        sum_data_size = sum(DSI_list[i])
        for j in range(n_classes):
            DSI_list[i][j] /= sum_data_size

    # print(DSI_list)
    #
    # for i in range(n_users):
    #     print(sum(DSI_list[i]))

    return idx_batch, DSI_list

def fetch_dataloader(params):
    """
    Fetch and return train/dev dataloader with hyperparameters (params.subset_percent = 1.)
    :param types:
    :param params:
    :return:
    """
    # Using random crops and horizontal flip for train set
    if params.t_augmentation == "yes":
        train_transformer = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
    else:
        # Data augmentation can be turned off
        train_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])

    # Transformer for dev set
    test_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    data_path = params.t_dataset_path + '/' + params.t_dataset_type
    if params.t_dataset_type == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=train_transformer)
        testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=test_transformer)
    elif params.t_dataset_type == 'emnist':
        trainset = torchvision.datasets.EMNIST(root=data_path, train=True, download=True, transform=train_transformer)
        testset = torchvision.datasets.EMNIST(root=data_path, train=False, download=True, transform=test_transformer)
    elif params.t_dataset_type == 'svhn':
        trainset = torchvision.datasets.SVHN(root=data_path, train=True, download=True, transform=train_transformer)
        testset = torchvision.datasets.SVHN(root=data_path, train=False, download=True, transform=test_transformer)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=params.t_batch_size, shuffle=True, num_workers=params.t_num_workers, pin_memory=params.t_cuda)
    testloader = torch.utils.data.DataLoader(testset, batch_size=params.t_batch_size, shuffle=False, num_workers=params.t_num_workers, pin_memory=params.t_cuda)

    return trainloader, testloader

def fetch_iid_dataloader(params):
    """
    Sample IID dataloader
    :param types:
    :param params:
    :return:
    """
    if params.t_augmentation == "yes":
        train_transformer = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
    else:
        train_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])

    test_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    data_path = params.t_dataset_path + '/' + params.t_dataset_type
    if params.t_dataset_type == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=train_transformer)
        testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=test_transformer)
    elif params.t_dataset_type == 'emnist':
        trainset = torchvision.datasets.EMNIST(root=data_path, train=True, download=True, transform=train_transformer)
        testset = torchvision.datasets.EMNIST(root=data_path, train=False, download=True, transform=test_transformer)
    elif params.t_dataset_type == 'svhn':
        trainset = torchvision.datasets.SVHN(root=data_path, train=True, download=True, transform=train_transformer)
        testset = torchvision.datasets.SVHN(root=data_path, train=False, download=True, transform=test_transformer)

    n_items = int(len(trainset)/params.n_users)
    trainset_users = {}
    trainset_dist = []
    idx = [i for i in range(len(trainset))]
    for i in range(params.n_users):
        trainset_users[i] = set(np.random.choice(idx, n_items, replace=False)) ## 비복원추출
        trainset_dist.append(trainset_users[i][1, :])
        idx = list(set(idx) - trainset_users[i])

    DSI_list = []
    for i in range(params.n_users):
        cnt = [1 for _ in range(10)]  # Laplacian correction
        for j in range(len(trainset_dist[i])):
            cnt[int(trainset_dist[i][j])] += 1
        cnt = np.array(cnt)
        cnt = cnt / sum(cnt)
        DSI_list.append(cnt)
        # print(cnt, end=" => ")
        # print(sum(cnt))

    trainloader = []
    for i in range(params.n_users):
        trainloader.append(torch.utils.data.DataLoader(DatasetSplit(trainset, trainset_users[i]), batch_size=params.t_batch_size, shuffle=True, num_workers=params.t_num_workers, pin_memory=params.t_cuda))

    testloader = torch.utils.data.DataLoader(testset, batch_size=params.t_batch_size, shuffle=False, num_workers=params.t_num_workers, pin_memory=params.t_cuda)

    return trainloader, testloader, DSI_list

def fetch_noniid_dataloader(params):
    """
    Sample non-IID dataloader
    :param types:
    :param params:
    :return:
    """
    if params.t_augmentation == "yes":
        train_transformer = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
    else:
        train_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])

    test_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    data_path = params.t_dataset_path + '/' + params.t_dataset_type
    if params.t_dataset_type == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=train_transformer)
        testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=test_transformer)
    elif params.t_dataset_type == 'emnist':
        trainset = torchvision.datasets.EMNIST(root=data_path, train=True, download=True, transform=train_transformer)
        testset = torchvision.datasets.EMNIST(root=data_path, train=False, download=True, transform=test_transformer)
    elif params.t_dataset_type == 'svhn':
        trainset = torchvision.datasets.SVHN(root=data_path, train=True, download=True, transform=train_transformer)
        testset = torchvision.datasets.SVHN(root=data_path, train=False, download=True, transform=test_transformer)

    # 50,000 training samples => (n_images) images/shard * (n_shard) shards
    # Shards are allocated by n_images
    n_shards = params.t_n_shards
    n_images = params.t_n_images
    idx_shard = [i for i in range(n_shards)]
    trainset_users = {i: np.array([]) for i in range(params.n_users)}
    trainset_dist = {i: np.array([]) for i in range(params.n_users)}
    idxs = np.arange(n_shards * n_images)
    images = trainset.data
    labels = trainset.targets

    # Sort labels
    # label별로 indexing
    # -> (index, label) pair들을 label 기준으로 정렬
    # -> pair들에서 index만 추려내기 (index는 data와 mapping 되어있음)
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    labels_dist = idxs_labels[1, :]

    # data_list = np.vstack((images, labels))
    # data_list = data_list[:, data_list[1, :].argsort()]

    # Divide and assign shards to clients
    for i in range(params.n_users):
        rand_set = set(np.random.choice(idx_shard, n_shards//params.n_users, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            trainset_users[i] = np.concatenate((trainset_users[i], idxs[rand * n_images:(rand+1) * n_images]), axis=0)
            trainset_dist[i] = np.concatenate((trainset_dist[i], labels_dist[rand * n_images:(rand+1) * n_images]), axis=0)

    DSI_list = []
    for i in range(params.n_users):
        cnt = [1 for _ in range(10)]  # Laplacian correction
        for j in range(len(trainset_dist[i])):
            cnt[int(trainset_dist[i][j])] += 1
        cnt = np.array(cnt)
        cnt = cnt / sum(cnt)
        DSI_list.append(cnt)
        # print(cnt, end=" => ")
        # print(sum(cnt))

    trainloader = []
    for i in range(params.n_users):
        trainloader.append(torch.utils.data.DataLoader(DatasetSplit(trainset, trainset_users[i]), batch_size=params.t_batch_size, shuffle=True, num_workers=params.t_num_workers, pin_memory=params.t_cuda))

    testloader = torch.utils.data.DataLoader(testset, batch_size=params.t_batch_size, shuffle=False, num_workers=params.t_num_workers, pin_memory=params.t_cuda)

    return trainloader, testloader, DSI_list

# def fetch_noniid_dirichlet_dataloader(params):
#     """
#         Sample non-IID dataloader
#         :param types:
#         :param params:
#         :return:
#         """
#     if params.t_augmentation == "yes":
#         train_transformer = transforms.Compose([
#             transforms.RandomCrop(32, padding=4),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
#         ])
#     else:
#         train_transformer = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
#         ])
#
#     test_transformer = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
#     ])
#     data_path = params.t_dataset_path + '/' + params.t_dataset_type
#     if params.t_dataset_type == 'cifar10':
#         trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=train_transformer)
#         testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=test_transformer)
#     elif params.t_dataset_type == 'emnist':
#         trainset = torchvision.datasets.EMNIST(root=data_path, train=True, download=True, transform=train_transformer)
#         testset = torchvision.datasets.EMNIST(root=data_path, train=False, download=True, transform=test_transformer)
#     elif params.t_dataset_type == 'svhn':
#         trainset = torchvision.datasets.SVHN(root=data_path, train=True, download=True, transform=train_transformer)
#         testset = torchvision.datasets.SVHN(root=data_path, train=False, download=True, transform=test_transformer)
#
#     num_classes = len(trainset.classes)
#     labels = trainset.targets
#     idxs = np.arange(len(trainset.data))
#
#     # Sort labels
#     # label별로 indexing
#     # -> (index, label) pair들을 label 기준으로 정렬
#     # -> pair들에서 index만 추려내기 (index는 data와 mapping 되어있음)
#     idxs_labels = np.vstack((idxs, labels))
#     idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
#     idxs = idxs_labels[0, :]
#
#     idxs_class = []
#
#     for i in range(10):
#         idxs_class.append(idxs[i * 5000:(i + 1) * 5000])
#
#     idx = [torch.where(torch.tensor(trainset.targets) == i) for i in range(num_classes)]
#     train_data = [trainset.data[idx[i][0]] for i in range(num_classes)]
#
#     s = np.random.dirichlet(np.ones(num_classes) * params.alpha, params.n_users)
#     data_dist = np.zeros((params.n_users, num_classes))
#     for i in range(params.n_users):
#         data_dist[i] = ((s[i] * len(train_data[0])).astype('int') / (s[i] * len(train_data[0])).astype('int').sum() * len(train_data[0])).astype('int')
#         data_num = data_dist[i].sum()
#         data_dist[i][np.random.randint(low=0, high=num_classes)] += ((len(train_data[0]) - data_num))
#         data_dist = data_dist.astype('int')
#
#     DSI_list = data_dist / (len(trainset.data) / params.n_users) # n_users * n_class
#
#     trainset_users = {i: np.array([]) for i in range(params.n_users)}
#
#     for i in range(params.n_users):
#         for j in range(num_classes):
#             if data_dist[i][j] != 0:
#                 d_index = np.random.randint(low=0, high=len(train_data[j]), size=data_dist[i][j]) # 0 ~ 5000까지에서 data_dist만큼 뽑음
#                 trainset_users[i] = np.concatenate((trainset_users[i], idxs_class[j][d_index]), axis=0)
#
#     trainloader = []
#
#     for i in range(params.n_users):
#         trainloader.append(torch.utils.data.DataLoader(DatasetSplit(trainset, trainset_users[i]), batch_size=params.t_batch_size, shuffle=True, num_workers=params.t_num_workers, pin_memory=params.t_cuda))
#
#     testloader = torch.utils.data.DataLoader(testset, batch_size=params.t_batch_size, shuffle=False, num_workers=params.t_num_workers, pin_memory=params.t_cuda)
#
#     return trainloader, testloader, DSI_list

def fetch_noniid_dirichlet_dataloader(params):
    """
        Sample non-IID dataloader
        :param types:
        :param params:
        :return:
        """
    if params.t_augmentation == "yes" and params.t_dataset_type == "cifar10":
        train_transformer = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        test_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
    elif params.t_augmentation == "yes" and params.t_dataset_type == "fmnist":
        train_transformer = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_transformer = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif params.t_augmentation == "yes" and params.t_dataset_type == "mnist":
        train_transformer = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.06078,), (0.1957,))
        ])
        test_transformer = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.06078,), (0.1957,))
        ])
    else:
        train_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        test_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
    data_path = params.t_dataset_path + '/' + params.t_dataset_type
    if params.t_dataset_type == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=train_transformer)
        testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=test_transformer)
    elif params.t_dataset_type == 'emnist':
        trainset = torchvision.datasets.EMNIST(root=data_path, train=True, download=True, transform=train_transformer)
        testset = torchvision.datasets.EMNIST(root=data_path, train=False, download=True, transform=test_transformer)
    elif params.t_dataset_type == 'svhn':
        trainset = torchvision.datasets.SVHN(root=data_path, train=True, download=True, transform=train_transformer)
        testset = torchvision.datasets.SVHN(root=data_path, train=False, download=True, transform=test_transformer)
    elif params.t_dataset_type == 'fmnist':
        trainset = torchvision.datasets.FashionMNIST(root=data_path, train=True, download=True, transform=train_transformer)
        testset = torchvision.datasets.FashionMNIST(root=data_path, train=False, download=True, transform=test_transformer)
    elif params.t_dataset_type == 'mnist':
        print(train_transformer)
        trainset = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=train_transformer)
        testset = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=test_transformer)

    n_classes = len(trainset.classes)
    if params.t_dataset_type == 'cifar10':
        idx_batch, DSI_list = getDirichletData(trainset, [1.0/params.n_users for _ in range(params.n_users)], params.alpha, n_classes)
    else:
        idx_batch, DSI_list = getDirichletData(trainset, [1.0 / params.n_users for _ in range(params.n_users)],
                                               params.alpha, n_classes)

    trainloader = []
    total_size = 0
    for i in range(params.n_users):
        trainloader.append(torch.utils.data.DataLoader(DatasetSplit(trainset, idx_batch[i]), batch_size=params.t_batch_size, shuffle=True, num_workers=params.t_num_workers, pin_memory=params.t_cuda))
        total_size += len(trainloader[i].dataset)

    testloader = torch.utils.data.DataLoader(testset, batch_size=params.t_batch_size, shuffle=False, num_workers=params.t_num_workers, pin_memory=params.t_cuda)
    print("Total data size: {}".format(total_size))

    return trainloader, testloader, np.array(DSI_list)

def fetch_noniid_classwise_dataloader(params):
    """
    Separate dataset by each class (total 10 fractions of data in CIFAR10)
    :param types:
    :param params:
    :return:
    """
    if params.t_augmentation == "yes" and params.t_dataset_type == "cifar10":
        train_transformer = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
    elif params.t_augmentation == "yes" and params.t_dataset_type == "fmnist":
        train_transformer = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif params.t_augmentation == "yes" and params.t_dataset_type == "mnist":
        train_transformer = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.06078,),(0.1957,))
        ])
    else:
        # Data augmentation can be turned off
        train_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])

    test_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    data_path = params.t_dataset_path + '/' + params.t_dataset_type
    if params.t_dataset_type == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=train_transformer)
        testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=test_transformer)
    elif params.t_dataset_type == 'emnist':
        trainset = torchvision.datasets.EMNIST(root=data_path, train=True, download=True, transform=train_transformer)
        testset = torchvision.datasets.EMNIST(root=data_path, train=False, download=True, transform=test_transformer)
    elif params.t_dataset_type == 'svhn':
        trainset = torchvision.datasets.SVHN(root=data_path, train=True, download=True, transform=train_transformer)
        testset = torchvision.datasets.SVHN(root=data_path, train=False, download=True, transform=test_transformer)

    trainset_users = {i:[] for i in range(10)}
    # print(trainset.data)
    for elem in trainset:
        trainset_users[elem[1]].append(elem)

    trainloader = []
    for i in range(10):
        trainloader.append(torch.utils.data.DataLoader(trainset_users[i], batch_size=params.t_batch_size, shuffle=True, num_workers=params.t_num_workers, pin_memory=params.t_cuda))

    testloader = torch.utils.data.DataLoader(testset, batch_size=params.t_batch_size, shuffle=False, num_workers=params.t_num_workers, pin_memory=params.t_cuda)

    return trainloader, testloader