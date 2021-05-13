import argparse
import os
import json
import shutil
import random
from itertools import islice

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


import torch.utils.data as data
from datasets import get_CIFAR10, get_SVHN, get_FashionMNIST, get_MNIST, postprocess
from model import Glow, load_model_from_param
from datasets import get_CIFAR10, get_SVHN, get_MNIST, get_FashionMNIST 
from utils_ood import *

from torch import autograd
from torchvision.utils import make_grid
from modules import (
    gaussian_sample,
)

from functools import partial
import torch.optim as optim
import tqdm
import numpy as np
import copy
import os





#### Another STATS from model :


def fischer_approximation_from_model(model, T = 1000, temperature = 1, type_fischer = "generated", sampling_dataset = None):
    fischer_matrix = None
    n = 0
    index = 0
    while n<T and index<len(sampling_dataset) :
        if index%100 == 0 :
            print(f"Index {index} on {len(sampling_dataset)}, n {n} on {T}")
        if type_fischer == "generated" :
            with torch.no_grad():
                mean, logs = model.prior(None,batch_size = 1)
                z = gaussian_sample(mean, logs, temperature = temperature)
                x = model.flow(z, temperature=temperature, reverse=True)[0]
        elif type_fischer == "sampled" :
            x = sampling_dataset[index][0].cuda()
            index+=1
        else :
            if n == 0:
                mean, logs = model.prior(None,batch_size = 1)
                z = gaussian_sample(mean, logs, temperature = temperature)
                x = model.flow(z, temperature=temperature, reverse=True)[0]
            else :
                return torch.ones(fischer_matrix.flatten().shape[0]).cuda()

        model.zero_grad()
        _, nll, _ = model(x.unsqueeze(0))
        nll.backward()
        current_grad = []
        for _, param in model.named_parameters():
            if param.grad is not None :
                current_grad.append(-param.grad.view(-1))

        current_grad = torch.cat(current_grad)**2
        if torch.isinf(current_grad).any() :
            continue
        if fischer_matrix is None :
            fischer_matrix = copy.deepcopy(current_grad)
        else :
            fischer_matrix = (1/n+1) * (n * current_grad + fischer_matrix)
        n+=1
    return fischer_matrix + 1e-8


def gradient_mean_from_model(model, sampling_dataset , T = 1000):
    total_grad = None
    n = 0
    index = 0
    while n<T and index< len(sampling_dataset) :
        if index%100 == 0 :
            print(f"Index {index} on {len(sampling_dataset)}, n {n} on {T}")
        model.zero_grad()
        x = sampling_dataset[index][0].cuda()
        index+=1

        _, nll, _ = model(x.unsqueeze(0))
        nll.backward()
        current_grad = []
        for _, param in model.named_parameters():
            if param.grad is not None :
                current_grad.append(-param.grad.view(-1))

        current_grad = torch.cat(current_grad)
        if torch.isinf(current_grad).any() :
            continue

        if total_grad is None :
            total_grad = copy.deepcopy(current_grad)
        else :
            total_grad = (1/n+1) * (n * current_grad + total_grad)
        n+=1
    # ** .75 : power to fischer matrix, 
    return total_grad



def log_p_data_from_model(model, sampling_dataset, mean_calculation_limit = 10000):
    with torch.no_grad():
        log_p = 0
        n = 0
        index = 0
        while index < mean_calculation_limit and index < len(sampling_dataset) :
            x = sampling_dataset[index][0].cuda()
            index+=1
            _, nll, _ = model(x.unsqueeze(0))
            log_p += -nll
        # ** .75 : power to fischer matrix, 
    return log_p/index

