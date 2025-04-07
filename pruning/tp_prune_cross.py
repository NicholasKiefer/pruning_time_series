import argparse
import json
import os
import torch
import torch_pruning as tp
from Crossformer.cross_exp.exp_crossformer import Exp_crossformer
from Crossformer.cross_models.attn import AttentionLayer
from typing import Sequence
import sys
import torchinfo
import numpy as np


class someargs(object):
    root_path = "./datasets/"
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
    exp.model.load_state_dict(torch.load(os.path.join(rootpath, "checkpoint.pth"), ))
    return exp

def nnz_print(model):
    nnz = 0
    params = 0
    for param in model.parameters():
        nnz += torch.count_nonzero(param)
        params += np.prod(list(param.shape))
    nnz_info = json.dumps({"nnz": int(nnz.item()), "density": float(nnz.item() / params), "params": int(params)})
    print(nnz_info)
    return nnz_info



args = someargs()
densities = [0.8 ** n for n in range(10)]

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

for model in os.listdir("checkpoints/"):
    args.setting = model
    args = parse_string_to_args(args)
    print(f"pred_len is {args.out_len}")
    print(f"dataset is {args.data}")
    print(f"modelid is {args.itr}")
    try:
        exp = load_exp(args)
    except RuntimeError:
        print(f"failed, {args.setting}")
        continue
    model = exp.model
    torchinfo.summary(model, )
    example_inputs = torch.randn([3, args.in_len, datasets[args.data][-1]], device=next(model.parameters()).device)

    for density in densities:

        print(f"{density=:.3f}")
        exp = load_exp(args)
        model = exp.model

        num_heads = {}
        for m in exp.model.modules():
            if isinstance(m, AttentionLayer):
                num_heads[m.query_projection] = m.n_heads
                num_heads[m.key_projection] = m.n_heads
                num_heads[m.value_projection] = m.n_heads
                # pass

        imp = tp.importance.MagnitudeImportance(target_types=[torch.nn.modules.conv._ConvNd, torch.nn.Linear, torch.nn.modules.batchnorm._BatchNorm, torch.nn.LayerNorm])
        # imp = tp.importance.RandomImportance()
        ignore = []
        for m in exp.model.modules():
            if isinstance(m, torch.nn.LayerNorm):
                ignore.append(m)
            if isinstance(m, torch.nn.Linear) and m.out_features == example_inputs.shape[-1]:
                ignore.append(m)
        for m in exp.model.decoder.decode_layers:
            ignore.append(m.linear_pred)

        pruner = tp.pruner.MetaPruner(
            exp.model,
            example_inputs,
            importance=imp,
            # iterative_steps=10,
            pruning_ratio=1-density, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
            root_module_types=[torch.nn.Conv1d, torch.nn.Linear, torch.nn.LayerNorm],
            ignored_layers=ignore,
            global_pruning=True,
            # max_pruning_ratio=0.5,
            # round_to=4,
            num_heads=num_heads,
            unwrapped_parameters=[(exp.model.enc_pos_embedding, -1), (exp.model.dec_pos_embedding, -1)],
            # customized_pruners={AutoCorrelationLayer: AttentionLayerPruner(),}
            )

        pruner.step()
        base_macs, base_nparams = tp.utils.count_ops_and_params(exp.model, example_inputs)
        # nnz = nnz_print(exp.model)
        
        metrics = exp.test(args.setting + f"_{density:.2f}ds")
        pruned_macs, pruned_params = tp.utils.count_ops_and_params(exp.model, example_inputs)
        run_fn = lambda mod, inp: mod(*inp)
        pruned_latency, _ = tp.utils.benchmark.measure_latency(exp.model, example_inputs, 500, 50, run_fn)
        print("Pruned Latency: {:.2f} ms, Pruned MACs: {:.2f} G, Pruned Params: {:.2f} M".format(pruned_latency, pruned_macs/1e9, pruned_params/1e6))
        with open("../results/result_tp_pruning.txt", "a") as file:
            file.write(json.dumps({"setting": args.setting + f"_{density:.2f}ds", "macs": base_macs, "params": base_nparams, "metrics": metrics}) + "\n")
        
        print("---")
