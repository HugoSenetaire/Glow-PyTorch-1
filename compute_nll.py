
import json

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from datasets import get_CIFAR10, get_SVHN, get_FashionMNIST, get_MNIST, postprocess
from model import Glow
from torch import autograd
from torchvision.utils import make_grid

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

def sample(model):
    with torch.no_grad():
        if hparams['y_condition']:
            y = torch.eye(num_classes)
            y = y.repeat(batch_size // num_classes + 1)
            y = y[:32, :].to(device) # number hardcoded in model for now
        else:
            y = None

        images = postprocess(model(y_onehot=y, temperature=1, reverse=True))

    return images.cpu()

def global_nlls(path, epoch, data1, data2, model, dataset1_name, dataset2_name, nb_step = 1, every_epoch = 10, lr = 1e-5):
    if epoch % every_epoch == 0 :
        lls1, grads1, statgrads1 = compute_nll(data1, model, nb_step = nb_step, lr = lr)
        lls2, grads2, statgrads2 = compute_nll(data2, model, nb_step = nb_step, lr = lr)




        output_path_global = os.path.join(path,"graphs")
        if not os.path.exists(output_path_global):
            os.makedirs(output_path_global)
        output_path_global = os.path.join(output_path_global, f"epoch{epoch}")
        if not os.path.exists(output_path_global):
            os.makedirs(output_path_global)

        output_image = sample(model)
        grid = make_grid(images[:30], nrow=6).permute(1,2,0)
        plt.figure(figsize=(10,10))
        plt.imshow(grid)
        plt.axis('off')
        plt.savefig(os.path.join(output_path_global,"samples.jpg"))


        save_figures(output_path_global, lls1, lls2, "log_likelihood", dataset1_name = dataset1_name, dataset2_name = dataset2_name)
        save_figures(output_path_global, grads1, grads2, "GRADS",dataset1_name = dataset1_name, dataset2_name = dataset2_name)
        save_figures(output_path_global, statgrads1, statgrads2, "STAT_GRADS",dataset1_name = dataset1_name, dataset2_name = dataset2_name)

        compute_roc_auc_scores(output_path_global, lls1, lls2, "log_likelihood")
        compute_roc_auc_scores(output_path_global, grads1, grads2, "GRADs")
        compute_roc_auc_scores(output_path_global, statgrads1, statgrads2, "statgrads")






def compute_nll(data, model, nb_step = 1, lr = 1e-5):
    torch.random.manual_seed(0)
    np.random.seed(0)
    
    
    lls = {}
    grad_total = {}
    grad_stat_total = {}
    for k in range(nb_step+1):
      lls[k] = []
      grad_total[k] = []
      grad_stat_total[k] = []



    for x in tqdm.tqdm(data) :
        # load weights.  print the weights.
        model_copy = copy.deepcopy(model).to(device_test)
        optimizer = optim.SGD(model_copy.parameters(), lr= lr, momentum = 0.)
        model_copy.zero_grad()


        grads = []
        diff_param = []
        x = x.to(device_test).unsqueeze(0)
        _, nll, _ = model_copy(x, y_onehot=None)
        nll.backward()
        lls[0].append(-nll.detach().cpu().item())
        optimizer.step()
        for name_copy, param_copy in model_copy.named_parameters():
            if param_copy.grad is not None :
                grads.append(-param_copy.grad.view(-1))
        grad_total[0].append(torch.sum(lr * (torch.cat(grads)**2)).detach().cpu().item())


        for (name_copy, param_copy), (name, param) in zip(model_copy.named_parameters(), model.named_parameters()):
            assert(name_copy == name)
            if param_copy.grad is not None :
                aux_diff_param = param_copy.data - param.data
                diff_param.append(aux_diff_param.view(-1))
        grads = torch.flatten(torch.cat(grads))
        diff_param = torch.flatten(torch.cat(diff_param))
        grad_stat_total[0].append(torch.abs(torch.dot(grads, diff_param)).detach().cpu().item())



        for k in range(1,nb_step+1):
            model_copy.zero_grad()
            grads = []
            diff_param = []
            _, nll, _ = model_copy(x, y_onehot=None)
            nll.backward()
            lls[k].append(-nll.detach().cpu().item())
            optimizer.step()
            for (name_copy, param_copy), (name, param) in zip(model_copy.named_parameters(), model.named_parameters()):
                assert(name_copy == name)
                if param_copy.grad is not None :
                    aux_diff_param = param_copy.data - param.data
                    diff_param.append(aux_diff_param.view(-1))
                    grads.append(-param_copy.grad.view(-1))
            grads = torch.flatten(torch.cat(grads))
            grad_total[k].append(torch.sum((grads **2)*lr).detach().cpu().item())
            diff_param = torch.flatten(torch.cat(diff_param))
            grad_stat_total[k].append(torch.abs(torch.dot(grads, diff_param)).detach().cpu().item())
           

    for key in grad_total.keys():
      grad_total[key] = np.array(grad_total[key])
      lls[key] = np.array(lls[key])
      grad_stat_total[key] = np.array(grad_stat_total[key])

        
    return lls, grad_total, grad_stat_total


def save_figures(output_path, input1, input2, prefix, dataset1_name, dataset2_name):
    for key in input1.keys():
        print(f"Steps {key}")
        print(f"{dataset2_name} {prefix}",np.mean(-input2[key]))
        print(f"{dataset1_name} {prefix}",np.mean(-input1[key]))


        plt.figure(figsize=(20,10))
        plt.title(f"Histogram Glow - trained on {dataset1_name}")
        plt.xlabel(f"{prefix}")
        plt.hist(input2[key], label=dataset2_name, density=True, bins=30, alpha = 0.8)
        plt.hist(input1[key], label=f"{dataset1_name}", density=True, bins=50, alpha =0.8)
        plt.legend()
        plt.savefig(os.path.join(output_path,f"{prefix}_Step{key}"))
        plt.close()

        plt.figure(figsize = (20,10))
        plt.boxplot([input2[key], input1[key]], labels = [dataset2_name, f"{dataset1_name}"])
        plt.savefig(os.path.join(output_path,f"{prefix}_BOXPLOT_Step{key}"))
        plt.close()


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
        test+= f"Result : {rocauc_score}, Result Reverse {1-rocauc_score} \n"

    with open(os.path.join(output_path, f"{prefix}_auroc"), "a") as f :
        f.writelines(test)


