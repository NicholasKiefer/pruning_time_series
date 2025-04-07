# load model, prune, then train test and save
import torch
import os
import json
import torch_pruning as tp
from typing import Sequence

from FEDformer.exp.exp_main import Exp_Main
from Crossformer.cross_exp.exp_crossformer import Exp_crossformer
from torch.nn.utils.prune import global_unstructured, L1Unstructured, remove
import sys


models = ["Transformer", "Informer", "Autoformer", "FEDformer", "Crossformer"]
pred_lens = [192]
densities = [.8 **n for n in range(10)]
density = .8 ** 5
datasets = {
    "ETTm1": ["ETT-small/", "ETTm1", "ETTm1.csv", 7],
}

class fedform_args(object):
    is_training = 1
    task_id = "test"
    model = "FEDformer"
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
    checkpoints = "FEDformer/checkpoints"
    seq_len = 96
    # seq_len = 720
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

class cross_args(object):
    root_path = "Crossformer/datasets/"
    data_split = [0.7,0.1,0.2]
    checkpoints = "Crossformer/checkpoints/"
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

def load_exp(args):
    if args.model == "Crossformer":
        exp = Exp_crossformer(args)
        rootpath = f"Crossformer/checkpoints/{args.setting}/"
        exp.model.load_state_dict(torch.load(os.path.join(rootpath, "checkpoint.pth"), ))
    else:
        rootpath = os.path.join(args.checkpoints, args.setting)
        # rootpath = f"checkpoints/{args.setting}/"
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


args = fedform_args()
args_ct = cross_args()

key, val = "ETTm1", ("ETT-small/", "ETTm1", "ETTm1.csv", 7)
print(f"dataset is {key}")
args.task_id = key
args.root_path = f"/data/Autoformer/{val[0]}"
args.data = val[1]
args.data_path = val[2]
args.enc_in = val[3]
args.dec_in = val[3]
args.c_out = val[3]
pred_len = 192
args.pred_len = pred_len
args.des = "Exp"
args.factor = 3

#  --data ETTm1 --in_len 672 --out_len 96 --seg_len 12 --learning_rate 1e-4 --itr 3 &
args_ct.root_path = f"/Crossformer/datasets/"
args_ct.data = val[1]
args_ct.data_dim = val[3]
args_ct.in_len = 672
args_ct.out_len = pred_len
args_ct.seg_len = 24
args_ct.win_size = 2
args_ct.d_model = 256
args_ct.n_heads = 4
args_ct.model = "Crossformer"
args_ct.data_path = val[2]



print(f"pred_len is {pred_len}")
for modelname in models:
    args.model = modelname
    args_ct = modelname
    print(f"modelid is {modelname}")
    for ii in range(3):
        if modelname == "Crossformer":
            args = args_ct
            setting = 'Crossformer_{}_il{}_ol{}_sl{}_win{}_fa{}_dm{}_nh{}_el{}_itr{}'.format(args.data, 
                args.in_len, args.out_len, args.seg_len, args.win_size, args.factor,
                args.d_model, args.n_heads, args.e_layers, ii)
            args.setting = setting
        else:
            print(f"model_exp_nmbr is {ii}")
            setting = '{}_{}_{}_modes{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_id, args.model, args.mode_select, args.modes, args.data, args.features, args.seq_len, args.label_len,
                args.pred_len, args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.factor, args.embed, args.distil, args.des, ii)
            args.setting = setting

        print(f"{density=:.3f}")


        exp = load_exp(args)        
        # pruning
        to_prune = []
        for mod in exp.model.modules():
            if isinstance(mod, torch.nn.Linear) and mod.out_features != 1 and mod.out_features != datasets[args.data][-1]:
                to_prune.append((mod, "weight"))
            if isinstance(mod, torch.nn.Conv1d) and mod.out_channels != datasets[args.data][-1]:
                to_prune.append((mod, "weight"))
        # print(to_prune)
        global_unstructured(to_prune, pruning_method=L1Unstructured, amount=1 - density)
        # for mod, name in to_prune:
                # remove(mod, "weight")
        
        # finetune
        exp.train(args.setting + "_finetune")

        metrics = exp.test(args.setting + f"_{density:.2f}ds")
        metrics["setting"] = args.setting
        metrics["density"] = f"{.8 ** 5:.2f}"
        with open("results/result_finetune.txt", "a") as file:
            file.write(json.dumps(metrics) + "\n")
