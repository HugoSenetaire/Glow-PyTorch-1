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



if (not torch.cuda.is_available()) :
    device_test = "cpu"
else :
    if torch.cuda.device_count()>1:
        device_test = "cuda:1"
    else :
        device_test = "cuda:0"

def sample(model):
    with torch.no_grad():

        y = None

        images = postprocess(model(y_onehot=y, temperature=1, reverse=True))

    return images.cpu()







def global_nlls(path, epoch, data1, data2, model, dataset1_name, dataset2_name, nb_step = 1, every_epoch = 10,optim_default = partial(optim.SGD, lr=1e-5, momentum = 0.), dataloader = False):
    if epoch % every_epoch == 0 :
        lls1, grads1, statgrads1, likelihood_ratio_statistic_1 = compute_nll(data1, model, nb_step = nb_step, optim_default = optim_default, dataloader = dataloader)
        lls2, grads2, statgrads2, likelihood_ratio_statistic_2 = compute_nll(data2, model, nb_step = nb_step, optim_default = optim_default, dataloader = dataloader)



        fischer_approximation_matrix = {}
        fischer_approximation_matrix[1000] = fischer_approximation(model)
        fischer_score_1 = calculate_score_statistic(data1, model, fischer_approximation_matrix, dataloader = dataloader)
        fischer_score_2 = calculate_score_statistic(data2, model, fischer_approximation_matrix, dataloader = dataloader) 

        output_path_global = os.path.join(path,"graphs")
        if not os.path.exists(output_path_global):
            os.makedirs(output_path_global)
        output_path_global = os.path.join(output_path_global, f"epoch{epoch}")
        if not os.path.exists(output_path_global):
            os.makedirs(output_path_global)

        output_image = sample(model)
        grid = make_grid(output_image[:30], nrow=6).permute(1,2,0)
        plt.figure(figsize=(10,10))
        plt.imshow(grid)
        plt.axis('off')
        plt.savefig(os.path.join(output_path_global,"samples.jpg"))


        save_figures(output_path_global, lls1, lls2, "log_likelihood", dataset1_name = dataset1_name, dataset2_name = dataset2_name)
        save_figures(output_path_global, grads1, grads2, "GRADS",dataset1_name = dataset1_name, dataset2_name = dataset2_name)
        save_figures(output_path_global, statgrads1, statgrads2, "STAT_GRADS",dataset1_name = dataset1_name, dataset2_name = dataset2_name)
        save_figures(output_path_global, likelihood_ratio_statistic_1, likelihood_ratio_statistic_2, "Likelihood ratio", dataset1_name = dataset1_name, dataset2_name = dataset2_name)
        save_figures(output_path_global, fischer_score_1, fischer_score_2, "Score Stats", dataset1_name = dataset1_name, dataset2_name = dataset2_name)

        compute_roc_auc_scores(output_path_global, lls1, lls2, "log_likelihood")
        compute_roc_auc_scores(output_path_global, grads1, grads2, "GRADs")
        compute_roc_auc_scores(output_path_global, statgrads1, statgrads2, "statgrads")
        compute_roc_auc_scores(output_path_global, likelihood_ratio_statistic_1, likelihood_ratio_statistic_2, "LikelihoodRatio")
        compute_roc_auc_scores(output_path_global, fischer_score_1, fischer_score_2, "FischerScoreStats")






def compute_nll(data, model, nb_step = 1, optim_default = partial(optim.SGD, lr=1e-5, momentum=0.), dataloader = False):
    torch.random.manual_seed(0)
    np.random.seed(0)
    

    lls = {}
    grad_total = {}
    grad_stat_total = {}
    likelihood_ratio_statistic = {}

    for k in range(nb_step+1):
      lls[k] = []
      grad_total[k] = []
      grad_stat_total[k] = []
      likelihood_ratio_statistic[k] = []

    if not dataloader :
        dataloader_aux = [tqdm.tqdm(data)]
    else :
        dataloader_aux = tqdm.tqdm(iter(dataloader))
    for data_list in dataloader_aux :
        for x in data_list :
            # load weights.  print the weights.
            model_copy = copy.deepcopy(model).to(device_test)
            optimizer = optim_default(model_copy.parameters())
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
                break

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

                grad_total[k].append(torch.sum((grads **2)*lr).detach().cpu().item())
                diff_param = torch.flatten(torch.cat(diff_param))
                grad_stat_total[k].append(torch.abs(torch.dot(grads, diff_param)).detach().cpu().item())
           

    for key in grad_total.keys():
      grad_total[key] = np.array(grad_total[key])
      lls[key] = np.array(lls[key])
      likelihood_ratio_statistic[key] = lls[key] - lls[0]
      grad_stat_total[key] = np.array(grad_stat_total[key])

        
    return lls, grad_total, grad_stat_total, likelihood_ratio_statistic


### Model with loading weights :


def global_nlls_from_model(path, epoch, data1, data2, model, dataset1_name, dataset2_name, pathmodel, image_shape, num_classes, nb_step = 1, every_epoch = 10, optim_default = partial(optim.SGD, lr=1e-5, momentum = 0.), dataloader = False):

    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), os.path.join(path,"current_tested_model.pth"))
    pathweights = os.path.join(path,"current_tested_model.pth")

    
    if epoch % every_epoch == 0 :
        lls1, grads1, statgrads1, likelihood_ratio_statistic_1 = compute_nll_from_model(data1, pathmodel, pathweights, image_shape, num_classes, nb_step = nb_step, optim_default = optim_default, dataloader=dataloader)
        lls2, grads2, statgrads2, likelihood_ratio_statistic_2 = compute_nll_from_model(data2, pathmodel, pathweights, image_shape, num_classes, nb_step = nb_step, optim_default = optim_default, dataloader=dataloader)


        output_path_global = os.path.join(path,"graphs")
        if not os.path.exists(output_path_global):
            os.makedirs(output_path_global)
        output_path_global = os.path.join(output_path_global, f"epoch{epoch}")
        if not os.path.exists(output_path_global):
            os.makedirs(output_path_global)

        output_image = sample(model)
        grid = make_grid(output_image[:30], nrow=6).permute(1,2,0)
        plt.figure(figsize=(10,10))
        plt.imshow(grid)
        plt.axis('off')
        plt.savefig(os.path.join(output_path_global,"samples.jpg"))


        save_figures(output_path_global, lls1, lls2, "log_likelihood", dataset1_name = dataset1_name, dataset2_name = dataset2_name)
        save_figures(output_path_global, grads1, grads2, "GRADS",dataset1_name = dataset1_name, dataset2_name = dataset2_name)
        save_figures(output_path_global, statgrads1, statgrads2, "STAT_GRADS",dataset1_name = dataset1_name, dataset2_name = dataset2_name)
        save_figures(output_path_global, likelihood_ratio_statistic_1, likelihood_ratio_statistic_2, "Likelihood ratio", dataset1_name = dataset1_name, dataset2_name = dataset2_name)

        compute_roc_auc_scores(output_path_global, lls1, lls2, "log_likelihood")
        compute_roc_auc_scores(output_path_global, grads1, grads2, "GRADs")
        compute_roc_auc_scores(output_path_global, statgrads1, statgrads2, "statgrads")
        compute_roc_auc_scores(output_path_global, likelihood_ratio_statistic_1, likelihood_ratio_statistic_2, "LikelihoodRatio")





def compute_nll_from_model(data, pathmodel, pathweights, image_shape, num_classes, nb_step = 5, optim_default = partial(optim.SGD, lr=1e-5, momentum=0.), dataloader = False):
    torch.random.manual_seed(0)
    np.random.seed(0)
    
    
    lls = {}
    grad_total = {}
    grad_stat_total = {}
    likelihood_ratio_statistic = {}



    for k in range(nb_step+1):
      lls[k] = []
      grad_total[k] = []
      grad_stat_total[k] = []
      likelihood_ratio_statistic[k] = []

    model = load_model_from_param(pathmodel, pathweights, num_classes, image_shape).cuda()

    if not dataloader :
        dataloader_aux = [tqdm.tqdm(data)]
    else :
        dataloader_aux = tqdm.tqdm(iter(dataloader))
    for data_list in dataloader_aux :
        for x in data_list :
            # load weights.  print the weights.
            model_copy = load_model_from_param(pathmodel, pathweights, num_classes, image_shape).cuda()
            optimizer = optim_default(model_copy.parameters())
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
                break

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
                grad_total[k].append(torch.sum((grads **2)*lr).detach().cpu().item())
                diff_param = torch.flatten(torch.cat(diff_param))
                grad_stat_total[k].append(torch.abs(torch.dot(grads, diff_param)).detach().cpu().item())
           

    for key in grad_total.keys():
      grad_total[key] = np.array(grad_total[key])
      lls[key] = np.array(lls[key])
      likelihood_ratio_statistic[key] = lls[key] - lls[0]
      grad_stat_total[key] = np.array(grad_stat_total[key])

        
    return lls, grad_total, grad_stat_total, likelihood_ratio_statistic


#### Another Stats :



def global_fischer_stats(path, epoch, data1, data2, model, dataset1_name, dataset2_name, nb_step = 1, T_list = [1000], every_epoch = 10, dataloader = False):
    if epoch % every_epoch == 0 :

        fischer_approximation_matrix = {}
        for T in T_list :
            fischer_approximation_matrix[T] = fischer_approximation(model)
        fischer_score_1 = calculate_score_statistic(data1, model, fischer_approximation_matrix, dataloader = dataloader)
        fischer_score_2 = calculate_score_statistic(data2, model, fischer_approximation_matrix, dataloader = dataloader) 

        output_path_global = os.path.join(path,"graphs")
        if not os.path.exists(output_path_global):
            os.makedirs(output_path_global)
        output_path_global = os.path.join(output_path_global, f"epoch{epoch}")
        if not os.path.exists(output_path_global):
            os.makedirs(output_path_global)

        save_figures(output_path_global, fischer_score_1, fischer_score_2, "Score Stats", dataset1_name = dataset1_name, dataset2_name = dataset2_name)
        compute_roc_auc_scores(output_path_global, fischer_score_1, fischer_score_2, "FischerScoreStats")




def fischer_approximation(model, T = 1000, temperature = 1):
    total_grad = None
    with torch.no_grad():
        mean, logs = model.prior(None,batch_size = T)
        z = gaussian_sample(mean, logs, temperature = temperature)
        list_img = model.flow(z, temperature=temperature, reverse=True)

    for x in list_img :
        model_copy = copy.deepcopy(model)
        model_copy.zero_grad()
        _, nll, _ = model_copy(x.unsqueeze(0))
        nll.backward()
        current_grad = []
        for name_copy, param_copy in model_copy.named_parameters():
            if param_copy.grad is not None :
                current_grad.append(-param_copy.grad.view(-1))

        current_grad = torch.cat(current_grad)**2
        if torch.isinf(current_grad).any() :
            T-=1
            continue
        if total_grad is None :
            total_grad = copy.deepcopy(current_grad)
        else :
            total_grad += current_grad
    

    return float(T)/(total_grad+1e-8)

def calculate_score_statistic(data, model, fischer_matrix, dataloader = False):
    torch.random.manual_seed(0)
    np.random.seed(0)
    score = {}
    for key in fischer_matrix.keys():
        score[key]= []

    if not dataloader :
        dataloader_aux = [tqdm.tqdm(data)]
    else :
        dataloader_aux = tqdm.tqdm(iter(dataloader))
    for data_list in dataloader_aux :
        for x in data_list :
            # load weights.  print the weights.
            model_copy = copy.deepcopy(model).to(device_test)
            model_copy.zero_grad()
            x = x.to(device_test).unsqueeze(0)
            grads = []

            _, nll, _ = model_copy(x, y_onehot=None)
            nll.backward()
            for name_copy, param_copy in model_copy.named_parameters():
                if param_copy.grad is not None :
                    grads.append(-param_copy.grad.view(-1))
            grads = torch.cat(grads)
            for key in fischer_matrix.keys():
                score[key].append(torch.mean(grads**2 * fischer_matrix[key]).detach().cpu().numpy())

    for key in fischer_matrix.keys():
        score[key] = np.array(score[key])

    return score




#### Another STATS from model :

def global_fisher_stat_from_model(path, epoch, data1, data2, model, dataset1_name, dataset2_name, pathmodel, image_shape, num_classes, T_list = [1000], every_epoch = 10, dataloader = False):

    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), os.path.join(path,"current_tested_model.pth"))
    pathweights = os.path.join(path,"current_tested_model.pth")

    
    if epoch % every_epoch == 0 :

        fischer_approximation_matrix = {}
        for T in T_list :
            fischer_approximation_matrix[T] = fischer_approximation_from_model(model, T=T)
        fischer_score_1 = calculate_score_statistic_from_model(data1, pathmodel, pathweights, fischer_approximation_matrix, image_shape, num_classes, dataloader = dataloader)
        fischer_score_2 = calculate_score_statistic_from_model(data2, pathmodel, pathweights, fischer_approximation_matrix, image_shape, num_classes, dataloader = dataloader) 

        output_path_global = os.path.join(path,"graphs")
        if not os.path.exists(output_path_global):
            os.makedirs(output_path_global)
        output_path_global = os.path.join(output_path_global, f"epoch{epoch}")
        if not os.path.exists(output_path_global):
            os.makedirs(output_path_global)

        save_figures(output_path_global, fischer_score_1, fischer_score_2, "Score Stats", dataset1_name = dataset1_name, dataset2_name = dataset2_name)
        compute_roc_auc_scores(output_path_global, fischer_score_1, fischer_score_2, "FischerScoreStats")


def fischer_approximation_from_model(model, T = 1000, temperature = 1):
    total_grad = None
    with torch.no_grad():
        mean, logs = model.prior(None,batch_size = T)
        z = gaussian_sample(mean, logs, temperature = temperature)
        list_img = model.flow(z, temperature=temperature, reverse=True)

    for x in list_img :
        model.zero_grad()
        _, nll, _ = model(x.unsqueeze(0))
        nll.backward()
        current_grad = []
        for name, param in model.named_parameters():
            if param.grad is not None :
                current_grad.append(-param.grad.view(-1))

        current_grad = torch.cat(current_grad)**2
        if torch.isinf(current_grad).any() :
            T-=1
            continue
        if total_grad is None :
            total_grad = copy.deepcopy(current_grad)
        else :
            total_grad += current_grad
    

    return float(T)/(total_grad+1e-8)

def calculate_score_statistic_from_model(data, pathmodel, pathweights, fischer_matrix, image_shape, num_classes, dataloader = False):
    torch.random.manual_seed(0)
    np.random.seed(0)
    score = {}
    for key in fischer_matrix.keys():
        score[key]= []

    if not dataloader :
        dataloader_aux = [tqdm.tqdm(data)]
    else :
        dataloader_aux = tqdm.tqdm(iter(dataloader))
    for data_list in dataloader_aux :
        for x in data_list :
            # load weights.  print the weights.
            model_copy = load_model_from_param(pathmodel, pathweights, num_classes, image_shape).cuda()
            x = x.to(device_test).unsqueeze(0)
            grads = []

            _, nll, _ = model_copy(x, y_onehot=None)
            nll.backward()
            for name_copy, param_copy in model_copy.named_parameters():
                if param_copy.grad is not None :
                    grads.append(-param_copy.grad.view(-1))
            grads = torch.cat(grads)
            for key in fischer_matrix.keys():
                score[key].append(torch.mean(grads**2 * fischer_matrix[key]).detach().cpu().numpy())

    for key in fischer_matrix.keys():
        score[key] = np.array(score[key])

    return score


### Utils general :


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

    with open(os.path.join(output_path, f"{prefix}_auroc.txt"), "a") as f :
        f.writelines(test)


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
    parser.add_argument("--T_list", nargs="+", default = [1000])

    

    args = parser.parse_args()
    # args = vars(args)
    
    device = torch.device("cuda")

    T_list = args.T_list
    for k in range(len(T_list)) :
        T_list[k] = int(T_list[k])


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

    model.load_state_dict(torch.load(checkpoint_path)['model'])
    model.set_actnorm_init()

    model = model.to(device)
    model = model.eval()

    if args.optim_type == "ADAM":
        optim_default = partial(optim.Adam, lr=args.lr_test)
    elif args.optim_type == "SGD":
        optim_default = partial(optim.SGD, lr = args.lr_test, momentum = args.momentum)

    if args.limited_data is not None :
        dataloader = False
        data1 = []
        data2 = []
        for k in range(args.limited_data):
            dataaux, targetaux = test_dataset[k]
            data1.append(dataaux)
            dataaux, targetaux = test_dataset_2[k]
            data2.append(dataaux)
    elif args.limited_data is None :
        dataloader = True
        data1 = data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=n_workers,
            drop_last=False,
        )
        data2 = data.DataLoader(
            test_dataset_2,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=n_workers,
            drop_last=False,
        )

    path = args.output_dir
    epoch = 1


    global_nlls(path, epoch, data1, data2, model, dataset1_name= args.dataset, dataset2_name=args.dataset2, nb_step = 1, every_epoch = 1, optim_default = optim_default, dataloader = dataloader)
    global_nlls_from_model(path, epoch, data1, data2, model, dataset1_name= args.dataset, dataset2_name=args.dataset2, pathmodel=params_path, image_shape=image_shape, num_classes=num_classes, nb_step = 1, every_epoch = 1, optim_default = optim_default, dataloader = dataloader)
    global_fischer_stats(path, epoch, data1, data2, model, dataset1_name= args.dataset, dataset2_name=args.dataset2, nb_step = 1, T_list = T_list, every_epoch = 1, dataloader = dataloader)
    global_fisher_stat_from_model(path, epoch, data1, data2, model, dataset1_name= args.dataset, dataset2_name=args.dataset2, pathmodel=params_path, image_shape=image_shape, num_classes=num_classes, T_list = T_list, every_epoch = 1, dataloader = dataloader)