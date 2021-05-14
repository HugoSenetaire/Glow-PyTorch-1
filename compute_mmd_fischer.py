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
from calculate_fischer_matrix import *


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


if (not torch.cuda.is_available()) :
    device_test = "cpu"
else :
    if torch.cuda.device_count()>1:
        device_test = "cuda:1"
    else :
        device_test = "cuda:0"





#### Another STATS from model :

def global_fisher_mmd_from_model(path, epoch, data1, data2, model, dataset1_name, dataset2_name, T_list_gradient= [1000], T_list_fischer = [1000], every_epoch = 10, dataloader = False, type_fischer = "generated", sampling_dataset = None):

    if not os.path.exists(path):
        os.makedirs(path)
   
    assert(len(T_list_fischer) == len(T_list_gradient))

    if epoch % every_epoch == 0 :
        inv_sqrt_fischer_approximation_matrix = {}
        gradient_mean = {}
        for T in T_list_fischer :
            inv_sqrt_fischer_approximation_matrix[T] = torch.pow(fischer_approximation_from_model(model, T=T, type_fischer=type_fischer, sampling_dataset=sampling_dataset), -1/2)
        for T in T_list_gradient : 
            gradient_mean[T] = gradient_mean_from_model(model, sampling_dataset, T)
        fischer_score_1 = calculate_fischer_mmd_from_model(data1, model, inv_sqrt_fischer_approximation_matrix, gradient_mean, dataloader = dataloader)
        fischer_score_2 = calculate_fischer_mmd_from_model(data2, model, inv_sqrt_fischer_approximation_matrix, gradient_mean, dataloader = dataloader) 

        output_path_global = os.path.join(path,"graphs")
        if not os.path.exists(output_path_global):
            os.makedirs(output_path_global)
        output_path_global = os.path.join(output_path_global, f"epoch{epoch}")
        if not os.path.exists(output_path_global):
            os.makedirs(output_path_global)

        save_figures(output_path_global, fischer_score_1, fischer_score_2, "MMD_Fischer", dataset1_name = dataset1_name, dataset2_name = dataset2_name)
        compute_roc_auc_scores(output_path_global, fischer_score_1, fischer_score_2, "MMD_Fischer")



def calculate_fischer_mmd_from_model(data, model, gradient_mean , inv_fischer_matrix_sqrt, dataloader = False):
    torch.random.manual_seed(0)
    np.random.seed(0)
    score = {}
    compteur = 0
    for key in zip(inv_fischer_matrix_sqrt.keys(), gradient_mean.keys()):
        score[key]= []

    if not dataloader :
        dataloader_aux = [(tqdm.tqdm(data),None)]
    else :
        dataloader_aux = tqdm.tqdm(iter(data))


    for data_list,_ in dataloader_aux :
        for x in data_list :
            model.zero_grad()
            # load weights.  print the weights.
            grads = []
            x = x.to(device_test).unsqueeze(0)
            _, nll, _ = model(x, y_onehot=None)
            nll.backward()
            for _, param_copy in model.named_parameters():
                # if param_copy.grad is not None and not torch.isinf(param_copy.grad).any() and not torch.isnan(param_copy.grad).any() :
                grads.append(-param_copy.grad.view(-1))
                # else : 
                    # print(name_copy)
            grads = torch.cat(grads)
            for key in zip(inv_fischer_matrix_sqrt.keys(), gradient_mean.keys()):
                score_aux = torch.linalg.norm( (grads - gradient_mean[key[1]]) * inv_fischer_matrix_sqrt[key[0]])
                if not torch.isinf(score_aux).any() and not torch.isnan(score_aux).any():
                    score[key].append(score_aux.detach().cpu().numpy())
                else :
                    compteur+=1
                    print("Error")

    for key in zip(inv_fischer_matrix_sqrt.keys(), gradient_mean.keys()):
        score[key] = np.array(score[key])
    
    print(f"Count of errors : {compteur}")")

    return score




if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "svhn", "mnist", "fashionmnist"],
        help="Type of the dataset to be used.",
    )

    parser.add_argument(
        "--dataset2",
        type=str,
        default="svhn",
        choices = ["cifar10", "svhn", "mnist", "fashionmnist"],
        help="Type of the dataset to be used for nlls comparisons",
    )

    parser.add_argument("--dataroot", type=str, help="path to dataset")
    parser.add_argument("--model_path", type=str, help = "path to model saved weights")
    parser.add_argument("--checkpoint", type = str)
    parser.add_argument("--download", action='store_true')

    parser.add_argument(
        "--batch_size", type=int, default=64, help="batch size used during training"
    )

    parser.add_argument(
        "--output_dir", type = str
    )

    parser.add_argument("--limited_data", type=int, default = None)



    parser.add_argument("--Nstep", type=int, default = 5)
    parser.add_argument("--T_list_fischer", nargs="+", default = [1000])
    parser.add_argument("--T_list_gradient", nargs="+", default = [1000])

    

    args = parser.parse_args()
    # args = vars(args)
    
    device = torch.device("cuda")

    T_list_gradient = args.T_list_gradient
    for k in range(len(T_list_gradient)) :
        T_list_gradient[k] = int(T_list_gradient[k])

    T_list_fischer = args.T_list_fischer
    for k in range(len(T_list_fischer)) :
        T_list_fischer[k] = int(T_list_fischer[k])


    model_path = args.model_path

    checkpoint_path = os.path.join(model_path, args.checkpoint)
    params_path = os.path.join(model_path, 'hparams.json')
    
    with open(os.path.join(model_path, 'hparams.json')) as json_file:  
        hparams = json.load(json_file)

    
        

    ds = check_dataset(args.dataset, args.dataroot, True, args.download)
    ds2 = check_dataset(args.dataset2, args.dataroot, True, args.download)
    image_shape, num_classes, train_dataset, test_dataset = ds
    image_shape2, num_classes2, train_dataset_2, test_dataset_2 = ds2

    


    model = Glow(image_shape, hparams['hidden_channels'], hparams['K'], hparams['L'], hparams['actnorm_scale'],
                hparams['flow_permutation'], hparams['flow_coupling'], hparams['LU_decomposed'], num_classes,
                hparams['learn_top'], hparams['y_condition'])
    
    dic = torch.load(checkpoint_path)
    if 'model' in dic.keys():
        model.load_state_dict(dic["model"])
    else :
        model.load_state_dict(dic)
    model.set_actnorm_init()

    model = model.to(device)
    model = model.eval()



    if args.limited_data is not None :
        dataloader = False
        data1 = []
        data2 = []

        indexes = np.arange(0, len(test_dataset), step=1)
        random.shuffle(indexes)
        indexes = indexes[:args.limited_data]
        for k in indexes:
            dataaux, targetaux = test_dataset[k]
            data1.append(dataaux)

        indexes = np.arange(0, len(test_dataset_2), step=1)
        random.shuffle(indexes)
        indexes = indexes[:args.limited_data]
        for k in indexes:
            dataaux, targetaux = test_dataset_2[k]
            data2.append(dataaux)

    elif args.limited_data is None :
        dataloader = True
        data1 = data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=6,
            drop_last=False,
        )
        data2 = data.DataLoader(
            test_dataset_2,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=6,
            drop_last=False,
        )
    path = args.output_dir
    epoch = 1

    path4 = os.path.join(path, "MMD_Fischer")
    global_fisher_mmd_from_model(path4+"_generated", epoch, data1, data2, model, dataset1_name= args.dataset, dataset2_name=args.dataset2, T_list_fischer = T_list_fischer, T_list_gradient = T_list_gradient, every_epoch = 1, dataloader = dataloader, type_fischer = "generated", sampling_dataset = test_dataset)
    global_fisher_mmd_from_model(path4+"_dataset", epoch, data1, data2, model, dataset1_name= args.dataset, dataset2_name=args.dataset2, T_list_fischer = T_list_fischer, T_list_gradient = T_list_gradient, every_epoch = 1, dataloader = dataloader, type_fischer = "sampled", sampling_dataset = test_dataset)
    global_fisher_mmd_from_model(path4+"_identity", epoch, data1, data2, model, dataset1_name= args.dataset, dataset2_name=args.dataset2, T_list_fischer = T_list_fischer, T_list_gradient = T_list_gradient, every_epoch = 1, dataloader = dataloader, type_fischer = "identity", sampling_dataset = test_dataset)