
import json

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from datasets import get_CIFAR10, get_SVHN, get_FashionMNIST, get_MNIST
from model import Glow
from torch import autograd

import tqdm
import numpy as np
import torch.optim as optim
import copy
import os

if (not torch.cuda.is_available()) :
    device_test = "cpu"
else :
    if torch.cuda.device_count()>1:
        device_test = "cuda:1"
    else :
        device_test = "cuda:0"


def global_nlls(path, epoch, data1, data2, model, dataset1_name = "CIFAR10", dataset2_name = "SVHN", nb_step = 1, every_epoch = 10):
    if epoch % every_epoch == 0 :
        nlls1, grads1 = compute_nll(data1, model, nb_step = nb_step)
        nlls2, grads2 = compute_nll(data2, model, nb_step = nb_step)


        output_path_global = os.path.join(path,"graphs")
        if not os.path.exists(output_path_global):
            os.makedirs(output_path_global)
        output_path_global = os.path.join(output_path_global, f"epoch{epoch}")
        if not os.path.exists(output_path_global):
            os.makedirs(output_path_global)


        save_nll(output_path_global, nlls1, nlls2, dataset1_name = "CIFAR10", dataset2_name = "SVHN")
        save_grad(output_path_global, grads1, grads2, dataset1_name = "CIFAR10", dataset2_name = "SVHN")

        compute_roc_auc_scores(output_path_global, nlls1, nlls2, "NLL")
        compute_roc_auc_scores(output_path_global, grads1, grads2, "GRADs")




def compute_nll(data, model, nb_step = 1):
    torch.random.manual_seed(0)
    np.random.seed(0)
    
    
    nlls = {}
    grad_total = {}
    for k in range(nb_step+1):
      nlls[k] = []
      grad_total[k] = []
    for x in tqdm.tqdm(data) :
        model_copy = copy.deepcopy(model).to(device_test)
        model_copy.zero_grad()
        optimizer = optim.Adam(model_copy.parameters(), lr=1e-5)
        x = x.to(device_test).unsqueeze(0)
        


        _, nll, _ = model_copy(x, y_onehot=None)
        nlls[0].append(nll.detach().cpu().item())


        for k in range(1,nb_step):
           nll.backward()
           grads = []
           for name, param in model_copy.named_parameters():
              if param.grad is not None :
                grads.append(param.grad.view(-1))
           grads = torch.sum(torch.cat(grads)**2).detach().cpu().item()
           grad_total[k].append(grads)


           optimizer.step()
           model_copy.zero_grad()
           _, nll, _ = model_copy(x, y_onehot=None)
           nlls[k+1].append(nll.detach().cpu().item())
          

        nll.backward()
        grads = []
        # for i,param in enumerate(model_copy_copy.parameters()):
        for name, param in model_copy.named_parameters():
          if param.grad is not None :
            grads.append(param.grad.view(-1))
        grads = torch.sum(torch.cat(grads)**2).detach().cpu().item()
        grad_total[nb_step].append(grads)

    for key in grad_total.keys():
      grad_total[key] = np.array(grad_total[key])
      nlls[key] = np.array(nlls[key])

    for key in grad_total.keys():
      grad_total[key] = np.array(grad_total[key])
      nlls[key] = np.array(nlls[key])

        
    return nlls, grad_total


def save_nll(output_path, nlls_1, nlls_2, dataset1_name = "CIFAR10", dataset2_name = "SVHN"):
    for key in nlls_1.keys():
        print(f"Steps {key}")
        print(f"{dataset2_name} NLL",np.mean(-nlls_2[key]))
        print(f"{dataset1_name} NLL",np.mean(-nlls_1[key]))


        plt.figure(figsize=(20,10))
        plt.title(f"Histogram Glow - trained on {dataset1_name}")
        plt.xlabel("Negative bits per dimension")
        plt.hist(-nlls_2[key], label=dataset2_name, density=True, bins=30, alpha = 0.8)
        plt.hist(-nlls_1[key], label=f"{dataset1_name}", density=True, bins=50, alpha =0.8)
        plt.legend()
        plt.savefig(os.path.join(output_path,f"NLLs_Step{key}"))
        plt.clf()

        plt.figure(figsize = (20,10))
        plt.boxplot([-nlls_2[key],-nlls_1[key]], labels = [dataset2_name, f"{dataset1_name}"])
        plt.savefig(os.path.join(output_path,f"NLLs_BOXPLOT_Step{key}"))
        plt.clf()


from sklearn.metrics import roc_auc_score
def compute_roc_auc_scores(output_path, list_1, list_2, prefix):
    test = ""
    for key in list_1.keys():
        test+= f"For step {key} \n "
        result_1 = list_1[key]
        label_1 = np.ones(np.shape(result_1))
        result_2 = list_2[key]
        label_2 = np.zeros(np.shape(result_2))
        label_total = np.concatenate((label_1, label_2))
        result_total = np.concatenate((result_1, result_2))
        rocauc_score = roc_auc_score(label_total, result_total)
        test+= f"Result : {rocauc_score}, Result Reverse {1-rocauc_score}"

    with open(os.path.join(output_path, f"{prefix}_auroc"), "a") as f :
        f.writelines(test)





def save_grad(output_path, grads_1, grads_2, dataset1_name = "CIFAR10", dataset2_name = "SVHN"):

    for key in grads_1.keys():

        print(f"Steps {key}")
        print(f"{dataset2_name} grad",np.mean(-grads_2[key]))
        print(f"{dataset1_name} grad",np.mean(-grads_1[key]))


        plt.figure(figsize=(20,10))
        plt.title("Histogram Glow - trained on CIFAR10")
        plt.xlabel("Negative bits per dimension")

        plt.hist(grads_2[key], label=dataset2_name, density=True, bins=30, alpha = 0.8)
        plt.hist(grads_1[key], label=dataset1_name, density=True, bins=50, alpha =0.8)
        plt.legend()
        plt.savefig(os.path.join(output_path,f"Grads_Step{key}"))
        plt.clf()


        plt.figure(figsize = (20,10))
        plt.boxplot([grads_2[key],grads_1[key]], labels = [dataset2_name, dataset1_name])
        plt.savefig(os.path.join(output_path,f"Grads_BOXPLOT_Step{key}"))
        plt.clf()