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

def global_typicality_mmd_from_model(path, epoch, data1, data2, model, dataset1_name, dataset2_name, sampling_dataset,  every_epoch = 10, dataloader = False, mean_calculation_limit = 1000000):

    if not os.path.exists(path):
        os.makedirs(path)

    
    if epoch % every_epoch == 0 :
        log_p_mean = log_p_data_from_model(model, sampling_dataset, mean_calculation_limit)
        fischer_score_1 = calculate_typicality_mmd_from_model(data1, model, log_p_mean, dataloader = dataloader)
        fischer_score_2 = calculate_typicality_mmd_from_model(data2, model, log_p_mean, dataloader = dataloader) 

        output_path_global = os.path.join(path,"graphs")
        if not os.path.exists(output_path_global):
            os.makedirs(output_path_global)
        output_path_global = os.path.join(output_path_global, f"epoch{epoch}")
        if not os.path.exists(output_path_global):
            os.makedirs(output_path_global)

        save_figures(output_path_global, fischer_score_1, fischer_score_2, "Typicality_mmd", dataset1_name = dataset1_name, dataset2_name = dataset2_name)
        compute_roc_auc_scores(output_path_global, fischer_score_1, fischer_score_2, "Typicality_mmd")




def calculate_typicality_mmd_from_model(data, model, log_p_mean, dataloader = False):
    torch.random.manual_seed(0)
    np.random.seed(0)
    score = {}
    for key in ["Typicality"]:
        score[key]= []

    if not dataloader :
        dataloader_aux = [(tqdm.tqdm(data),None)]
    else :
        dataloader_aux = tqdm.tqdm(iter(data))


    for data_list,_ in dataloader_aux :
        for x in data_list :
            model.zero_grad()
            x = x.to(device_test).unsqueeze(0)
            _, nll, _ = model(x, y_onehot=None)

            for key in score.keys():
                score[key].append((log_p_mean * nll).cpu().detach().numpy())
                      
    for key in score.keys():
        score[key] = np.array(score[key]).reshape(-1)

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
    parser.add_argument("--lr_test", type=float, default = 1e-5, help="Learning rate for the evaluation of ood")


    parser.add_argument("--optim_type", type=str, choices = ["ADAM", "SGD"])
    parser.add_argument("--momentum", type = float, default = 0.)

    parser.add_argument("--Nstep", type=int, default = 5)

    parser.add_argument("--mean_calculation_limit", type=int, default = 10000)

    

    args = parser.parse_args()
    # args = vars(args)
    
    device = torch.device("cuda")



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

    if args.optim_type == "ADAM":
        optim_default = partial(optim.Adam, lr=args.lr_test)
    elif args.optim_type == "SGD":
        optim_default = partial(optim.SGD, lr = args.lr_test, momentum = args.momentum)


    dataloader1 = data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=6,
            drop_last=False,
        )
    dataloader2 = data.DataLoader(
            test_dataset_2,
            batch_size=1,
            shuffle=True,
            num_workers=6,
            drop_last=False,
        )

    if args.limited_data is not None :
        dataloader = False

        data1 = []
        data2 = []
        iter1 = iter(dataloader1)
        iter2 = iter(dataloader2)
        for k in range(args.limited_data):
            dataaux, targetaux = next(iter1)
            data1.append(dataaux)
            dataaux, targetaux = next(iter2)
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

    path4 = os.path.join(path, "TypicalityMMD")

    global_typicality_mmd_from_model(path, epoch, data1, data2, model, dataset1_name= args.dataset, dataset2_name=args.dataset2, sampling_dataset = dataloader1,  every_epoch = 1, dataloader = False, mean_calculation_limit=args.mean_calculation_limit)