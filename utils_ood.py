import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
import os
import torch 
import torch.utils.data as data
from datasets import get_CIFAR10, get_SVHN, get_FashionMNIST, get_MNIST, postprocess
from model import Glow, load_model_from_param
from datasets import get_CIFAR10, get_SVHN, get_MNIST, get_FashionMNIST 


from torch import autograd
from torchvision.utils import make_grid
from modules import (
    gaussian_sample,
)


def sample(model):
    with torch.no_grad():

        y = None

        images = postprocess(model(y_onehot=y, temperature=1, reverse=True))

    return images.cpu()

def check_manual_seed(seed):
    seed = seed or random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)

    print("Using seed: {seed}".format(seed=seed))


def check_dataset(dataset, dataroot, augment, download):
    if dataset == "cifar10":
        cifar10 = get_CIFAR10(augment, dataroot, download)
        input_size, num_classes, train_dataset, test_dataset = cifar10
    if dataset == "svhn":
        svhn = get_SVHN(augment, dataroot, download)
        input_size, num_classes, train_dataset, test_dataset = svhn
    if dataset == "mnist":
        mnist = get_MNIST(augment, dataroot, download)
        input_size, num_classes, train_dataset, test_dataset = mnist
    if dataset == "fashionmnist":
        fashionmnist = get_FashionMNIST(augment, dataroot, download)
        input_size, num_classes, train_dataset, test_dataset = fashionmnist

    return input_size, num_classes, train_dataset, test_dataset

def save_figures(output_path, input1, input2, prefix, dataset1_name, dataset2_name):
    for key in input1.keys():
        
        print(f"Steps {key}")

        print(f"{dataset1_name} {prefix}",np.mean(-input1[key]))
        print(f"{dataset2_name} {prefix}",np.mean(-input2[key]))

        plt.figure(figsize=(20,10))
        plt.title(f"Histogram Glow - trained on {dataset1_name}")
        plt.xlabel(f"{prefix}")
        plt.hist(input1[key], label=f"{dataset1_name}", density=True, bins=50, alpha =0.8)
        plt.hist(input2[key], label=dataset2_name, density=True, bins=30, alpha = 0.8)
        plt.legend()
        plt.savefig(os.path.join(output_path,f"{prefix}_Step{key}"))
        plt.close()

        plt.figure(figsize = (20,10))
        print(input1[key])
        plt.boxplot([input1[key], input2[key]], labels = [dataset1_name, f"{dataset2_name}"])
        plt.savefig(os.path.join(output_path,f"{prefix}_BOXPLOT_Step{key}"))
        plt.close()


def compute_roc_auc_scores(output_path, list_1, list_2, prefix):
    test = ""
    for key in list_1.keys():
        test+= f"For step {key} \n "
        result_1 = list_1[key]
        if np.isinf(result_1).any() or np.isnan(result_1).any():
            print(f"Inf in the result for {prefix} step {key}")
            test+= f"Inf in the result \n"
            continue
       

        label_1 = np.ones(np.shape(result_1))
        result_2 = list_2[key]
        if np.isinf(result_2).any() or np.isnan(result_2).any():
            print(f"Inf in the result for {prefix} step {key}")
            test+= f"Inf in the result \n"
            continue
        label_2 = np.zeros(np.shape(result_2))
        label_total = np.concatenate((label_1, label_2))
        result_total = np.concatenate((result_1, result_2))
        rocauc_score = roc_auc_score(label_total, result_total)
        test+= f"Result : {rocauc_score}, Result Reverse {1-rocauc_score} \n"

    with open(os.path.join(output_path, f"{prefix}_auroc.txt"), "a") as f :
        f.writelines(test)