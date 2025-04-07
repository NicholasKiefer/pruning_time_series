import numpy as np
import json
import os
import sys
import torch
from Crossformer.cross_models.cross_former import Crossformer
from torch.nn.utils.prune import global_unstructured, L1Unstructured, remove

import torchmetrics
import torch
import torchvision
from torch.utils.data import DataLoader
from Crossformer.cross_exp.exp_crossformer import Exp_crossformer

import argparse
import torch
import torch_pruning as tp
import torchinfo

class someargs(object):
    root_path = "/datasets/"
    data_split = [0.7,0.1,0.2]
    checkpoints = f"{os.getcwd()}/../Crossformer/checkpoints/"
    d_ff = 512
    e_layers = 3
    dropout = 0.2
    baseline = False
    num_workers = 0
    batch_size = 32
    train_epochs = 20
    patience = 3
    learning_rate = 1e-4
    lradj = "type1"
    save_pred = False
    use_gpu = True
    gpu = 0
    use_multi_gpu = False
    factor = 10


def parse_string_to_args(args):
    with open(f"{args.checkpoints}/{args.setting}/args.json", "rb") as file:
        x = json.load(file)
    for k, v in x.items():
        setattr(args, k, v)
    return args


def load_exp(args):
    rootpath = f"{args.checkpoints}/{args.setting}/"
    exp = Exp_crossformer(args)
    # print(rootpath)
    exp.model.load_state_dict(torch.load(os.path.join(rootpath, "checkpoint.pth"), ))
    return exp

def nnz_print(model):
    nnz = 0
    params = 0
    for param in model.parameters():
        nnz += torch.count_nonzero(param)
        params += np.prod(list(param.shape))
    print(json.dumps({"nnz": int(nnz.item()), "density": float(nnz.item() / params), "params": int(params)}))
    # print(f"{nnz.item()=}, {params=}, density={nnz / params}")

args = someargs()
densities = [0.8 ** n for n in range(10)][1:]

datasets = {
    "ETTm1": ["ETTm1", "ETTm1.csv", 7],
    "ETTh1": ["ETTh1", "ETTh1.csv", 7],
    "ETTm2": ["ETTm2", "ETTm2.csv", 7],
    "ETTh2": ["ETTh2", "ETTh2.csv", 7],
    "ECL": ["custom", "ECL.csv", 321],
    "Exchange": ["custom", "exchange_rate.csv", 8],
    "Traffic": ["custom", "traffic.csv", 862],
    "weather": ["weather/", "custom", "weather.csv", 21],
    "weather": ["custom", "weather.csv", 21],
    
    }
already_done = open("../results/results_nn_pruning.txt", "r").readlines()
already_done = list(map(json.loads, already_done))
already_done = {i["setting"]: i for i in already_done}

for model in os.listdir("checkpoints/"):
    # args = someargs()
    args.setting = model
    args = parse_string_to_args(args)
    print(f"pred_len is {args.out_len}")
    print(f"dataset is {args.data}")
    print(f"modelid is {args.itr}")
    if args.setting + "_1.00ds" in already_done.keys():
        # print("already pruned")
        continue
    try:
        exp = load_exp(args)
    except RuntimeError:
        print(f"failed, {args.setting}")
        continue
    model = exp.model
    # print how many parameters are zero
    # print(list(model.modules())[0])
    torchinfo.summary(model, )
    nnz_print(model)

    exp.test(args.setting + "_1.00ds")
    print("---")

    for density in densities:
        if args.setting + f"_{density:.2f}ds" in already_done.keys():
            print("density already pruned")
            continue
        print(f"{density=:.3f}")
        exp = load_exp(args)
        model = exp.model
        
        # pruning
        to_prune = []
        for mod in model.modules():
            if isinstance(mod, torch.nn.Linear) and mod.out_features != 1 and mod.out_features != datasets[args.data][-1]:
                to_prune.append((mod, "weight"))
            if isinstance(mod, torch.nn.Conv1d) and mod.out_channels != datasets[args.data][-1]:
                to_prune.append((mod, "weight"))
        # print(to_prune)
        global_unstructured(to_prune, pruning_method=L1Unstructured, amount=1 - density)
        for mod, name in to_prune:
                remove(mod, "weight")
        # torchinfo.summary(model, )
        nnz_print(model)
        
        metrics = exp.test(args.setting + f"_{density:.2f}ds")
        metrics["setting"] = args.setting + f"_{density:.2f}ds"
        with open("../results/result_nn_pruning.txt", "a") as file:
            file.write(json.dumps(metrics) + "\n")
        print("---")
