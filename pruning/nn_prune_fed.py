import numpy as np
import json
import os
import torch
# from models import FEDformer, Autoformer, Informer, Transformer
from torch.nn.utils.prune import global_unstructured, L1Unstructured, remove

import torchmetrics
import torch
import torchvision
from torch.utils.data import DataLoader
from FEDformer.exp.exp_main import Exp_Main

import argparse
import torch
import torch_pruning as tp
import torchinfo
import sys

class someargs(object):
    
    is_training = 1
    task_id = "test"
    model = "Autoformer"
    version = "Fourier"
    mode_select = "random"
    modes = 64
    L = 3
    base = "legendre"
    cross_activation = "tanh"
    data = "ETTh1"
    root_path = "./dataset/ETT/"
    data_path = "ETTh1.csv"
    features = "M"
    target = "OT"
    freq = "h"
    detail_freq = "h"
    checkpoints = f"{os.getcwd()}/../FEDformer/checkpoints"
    seq_len = 96
    label_len = 48
    pred_len = 96
    enc_in = 7
    dec_in = 7
    c_out = 7
    d_model = 512
    n_heads = 8
    e_layers = 2
    d_layers = 1
    d_ff = 2048
    moving_avg = [24]
    factor = 1
    distil = True
    dropout = 0.05
    embed = "timeF"
    activation = "gelu"
    output_attention = False
    do_predict = False
    num_workers = 10
    itr = 3
    train_epochs = 10
    batch_size = 32
    patience = 3
    learning_rate = 0.0001
    des = "test"
    loss = "mse"
    lradj = "type1"
    use_amp = False
    use_gpu = True
    gpu = 0
    use_multi_gpu = False
    devices = "0"


def load_exp(args):
    rootpath = f"{args.checkpoints}/{args.setting}/"
    exp = Exp_Main(args)
    # print(rootpath)
    exp.model.load_state_dict(torch.load(os.path.join(rootpath, "checkpoint.pth"), ))
    if args.model == "FEDformer":
        modes = json.loads(open(os.path.join(rootpath, "modes.pt"), "r").read())
        exp.model.encoder.attn_layers[0].attention.inner_correlation.index = modes["encoder.attn_layers[0].attention.inner_correlation.index"]
        exp.model.encoder.attn_layers[1].attention.inner_correlation.index = modes["encoder.attn_layers[1].attention.inner_correlation.index"]
        exp.model.decoder.layers[0].self_attention.inner_correlation.index = modes["decoder.layers[0].self_attention.inner_correlation.index"]
        exp.model.decoder.layers[0].cross_attention.inner_correlation.index_q = modes["decoder.layers[0].cross_attention.inner_correlation.index_q"]
        exp.model.decoder.layers[0].cross_attention.inner_correlation.index_kv = modes["decoder.layers[0].cross_attention.inner_correlation.index_kv"]
        
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
args.root_path = "./data/Autoformer/ETT-small/"
args.factor = 3
# args.pred_len = 192
# args.task_id = "ETTh1"
args.features = "M"
args.des = "Exp"
# ratios = np.logspace(-.2, -3, 10)
densities = [0.8 ** n for n in range(10)][1:]

datasets = {
    "ETTm1": ["ETT-small/", "ETTm1", "ETTm1.csv", 7],
    "ETTh1": ["ETT-small/", "ETTh1", "ETTh1.csv", 7],
    "ETTm2": ["ETT-small/", "ETTm2", "ETTm2.csv", 7],
    "ETTh2": ["ETT-small/", "ETTh2", "ETTh2.csv", 7],
    "ECL": ["electricity/", "custom", "electricity.csv", 321],
    "Exchange": ["exchange_rate/", "custom", "exchange_rate.csv", 8],
    "traffic": ["traffic/", "custom", "traffic.csv", 862],
    "weather": ["weather/", "custom", "weather.csv", 21],
    
    }
already_done = open("../results/result_nn_pruning.txt", "r").readlines()
already_done = list(map(json.loads, already_done))
already_done = {i["setting"]: i for i in already_done}

for key, val in datasets.items():
    print(f"dataset is {key}")
    args.task_id = key
    args.root_path = f"../data/Autoformer/{val[0]}"
    args.data = val[1]
    args.data_path = val[2]
    args.enc_in = val[3]
    args.dec_in = val[3]
    args.c_out = val[3]
    for pred_len in [96, 192, 336, 720]:
        args.pred_len = pred_len
        print(f"pred_len is {pred_len}")
        for modelname in ["Transformer", "Informer", "Autoformer", "FEDformer"]:

            args.model = modelname
            print(f"modelid is {modelname}")
            for ii in range(3):
                print(f"model_exp_nmbr is {ii}")
                args.setting = '{}_{}_{}_modes{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                                args.task_id, args.model, args.mode_select, args.modes, args.data, args.features, args.seq_len,
                                args.label_len, args.pred_len, args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff,
                                args.factor, args.embed, args.distil, args.des, ii)
                try:
                    exp = load_exp(args)
                except FileNotFoundError:
                    print("model not found")
                    print(args.setting)
                    # pass
                    continue
                model = exp.model

                torchinfo.summary(model, )
                nnz_print(model)
                if torch.any(exp.model.enc_embedding.value_embedding.tokenConv.weight.isnan()):
                    print("model has NaN weights, skipping all densities")
                    print("---")
                    continue
                metrics = exp.test(args.setting + "_1.00ds")
                metrics["setting"] = args.setting + "_1.00ds"
                with open("result_pruning.txt", "a") as file:
                    file.write(json.dumps(metrics) + "\n")
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
                        if isinstance(mod, torch.nn.Linear) and mod.out_features != 1 and mod.out_features != 7:
                            to_prune.append((mod, "weight"))
                        if isinstance(mod, torch.nn.Conv1d) and mod.out_channels != 7:
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
