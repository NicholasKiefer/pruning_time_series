import argparse
import json
import os
import torch
import torch_pruning as tp
from FEDformer.layers.AutoCorrelation import AutoCorrelationLayer
from FEDformer.layers.Autoformer_EncDec import series_decomp_multi, series_decomp, moving_avg, EncoderLayer
from FEDformer.exp.exp_main import Exp_Main
from typing import Sequence

class someargs(object):
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

class AttentionLayerPruner(tp.BasePruningFunc):
    # TARGET_MODULES = ops.TORCH_LINEAR
    def __init__(self, pruning_dim=1):
        super().__init__(pruning_dim)
        self.ref = AutoCorrelationLayer
        self.shape = None
    def prune_out_channels(self, layer: AutoCorrelationLayer, idxs: Sequence[int]):
        # print(f"{layer.query_projection.weight.shape=}")
        tp.prune_linear_out_channels(layer.out_projection, idxs)
        tp.prune_linear_in_channels(layer.out_projection, idxs)
        tp.prune_linear_out_channels(layer.query_projection, idxs)
        tp.prune_linear_out_channels(layer.key_projection, idxs)
        tp.prune_linear_out_channels(layer.value_projection, idxs)
        # print(f"{layer.query_projection.weight.shape=}")
        if hasattr(layer, "inner_correlation"):
            if hasattr(layer.inner_correlation, "weights1"):
                # print(f"{layer.inner_correlation.weights1.data.shape=}")
                
                if 64 == layer.inner_correlation.weights1.data.shape[1]:
                
                    # small_idxs = set([i for i in idxs if i<40])
                    keep_idxs = list(set(range(layer.inner_correlation.weights1.data.shape[2])) - set(idxs))
                    keep_idxs.sort()
                    layer.inner_correlation.weights1 = self._prune_parameter_and_grad(layer.inner_correlation.weights1, keep_idxs, 2)
                    layer.inner_correlation.weights1 = self._prune_parameter_and_grad(layer.inner_correlation.weights1, keep_idxs, 1)
                    # print(f"{layer.inner_correlation.weights1.data.shape=}")
        return layer

    def prune_in_channels(self, layer: AutoCorrelationLayer, idxs):
        tp.prune_linear_in_channels(layer.query_projection, idxs)
        tp.prune_linear_in_channels(layer.key_projection, idxs)
        tp.prune_linear_in_channels(layer.value_projection, idxs)        
        return layer

    def get_out_channels(self, layer: torch.nn.Module):
        # print(f"got out channels for {layer}")
        return layer.out_projection.out_features

    def get_in_channels(self, layer: torch.nn.Module):
        # print("got in channels")
        # assert layer.query_projection.in_features == layer.key_projection.in_features
        return layer.query_projection.in_features
    
    def get_in_channel_groups(self, layer):
        return 8
    
    def get_out_channel_groups(self, layer):
        return 8


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


args = someargs()
args.root_path = "/hkfs/work/workspace_haic/scratch/vq6575-prune_nets/data/Autoformer/ETT-small/"
args.factor = 3
# args.pred_len = 192
# args.label_len = 48
# args.seq_len = 96
args.use_gpu = True if torch.cuda.is_available() else False
# args.task_id = "ETTh1"
args.des = "Exp"
args.features = "M"
# args.moving_avg = 24
# densities = [0.8 ** n for n in range(10)]

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

import sys
models = ["Transformer", "Informer", "Autoformer", "FEDformer"]
pred_lens = [96, 192, 336, 720]
with open("../results/result_pruning_tp.txt", "r") as file:
    already_done = file.readlines()
already_done = list(map(json.loads, already_done))
already_done = {i["setting"]: i for i in already_done}

for key, val in datasets.items():
    print(f"dataset is {key}")
    args.task_id = key
    args.root_path = f"/data/Autoformer/{val[0]}"
    args.data = val[1]
    args.data_path = val[2]
    args.enc_in = val[3]
    args.dec_in = val[3]
    args.c_out = val[3]
    for pred_len in pred_lens:
        args.pred_len = pred_len
        print(f"pred_len is {pred_len}")
        for modelname in models:
            args.model = modelname
            print(f"modelid is {modelname}")
            for ii in range(3):
                print(f"model_exp_nmbr is {ii}")
                setting = '{}_{}_{}_modes{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                    args.task_id, args.model, args.mode_select, args.modes, args.data, args.features, args.seq_len, args.label_len,
                    args.pred_len, args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.factor, args.embed, args.distil, args.des, ii)
                args.setting = setting
                try:
                    exp = load_exp(args)
                except FileNotFoundError:
                    print(f"model not found for setting {args.setting}")
                    # pass
                    continue
                model = exp.model
                # exit if model params contain nans (first layer should be enough)
                if torch.any(exp.model.enc_embedding.value_embedding.tokenConv.weight.isnan()):
                    print("model has NaN weights, skipping all densities")
                    print("---")
                    continue
                
                
                enc = torch.randn([3, args.seq_len, val[3]])
                enc_mark = torch.randn([3, args.seq_len, 4])
                dec = torch.randn([3, args.seq_len//2+args.pred_len, val[3]])
                dec_mark = torch.randn([3, args.seq_len//2+args.pred_len, 4])
                example_inputs = [enc, enc_mark, dec, dec_mark]
                if args.use_gpu:
                    example_inputs = [i.cuda() for i in example_inputs]
                    exp = load_exp(args)
                    imp = tp.importance.MagnitudeImportance(target_types=[torch.nn.modules.conv._ConvNd, torch.nn.Linear, torch.nn.modules.batchnorm._BatchNorm, torch.nn.LayerNorm, AutoCorrelationLayer])
                    ignore = []
                    num_heads = {}
                    for m in exp.model.modules():
                        if isinstance(m, torch.nn.Linear):
                            if m.out_features == val[3] or m.out_features == 1:
                                ignore.append(m)
                        if isinstance(m, torch.nn.Conv1d) and m.out_channels == val[3]:
                            ignore.append(m)
                    num_heads[exp.model.encoder.attn_layers[0].attention] = 8
                    num_heads[exp.model.encoder.attn_layers[0].attention.query_projection] = 8
                    num_heads[exp.model.encoder.attn_layers[1].attention] = 8
                    num_heads[exp.model.encoder.attn_layers[1].attention.query_projection] = 8
                    num_heads[exp.model.decoder.layers[0].self_attention.query_projection] = 8
                    num_heads[exp.model.decoder.layers[0].cross_attention.query_projection] = 8

                    pruner = tp.pruner.MetaPruner(
                        exp.model,
                        example_inputs,
                        importance=imp,
                        iterative_steps=50,
                        pruning_ratio=1., # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
                        root_module_types=[torch.nn.Conv1d, AutoCorrelationLayer, series_decomp_multi, torch.nn.Linear],
                        ignored_layers=ignore,
                        global_pruning=True,
                        unwrapped_parameters=[
                            ],
                        num_heads=num_heads,
                        customized_pruners={AutoCorrelationLayer: AttentionLayerPruner(),
                                            },
                        )

                    for it in range(50):
                        pruner.step()
                        run_fn = lambda mod, inp: mod(*inp)
                        pruned_macs, pruned_nparams = tp.utils.count_ops_and_params(pruner.model, example_inputs)
                        pruned_latency, _ = tp.utils.benchmark.measure_latency(exp.model, example_inputs, 300, 50, run_fn)
                        metrics = exp.test(setting + f"_{it}it")
                        res = json.dumps({"setting": args.setting + f"_{it}it", "latency": pruned_latency, "nparams": pruned_nparams, "macs": pruned_macs, "nparams": pruned_nparams, "metrics": metrics})
                        print(res)
                        with open("../results/result_tp_pruning.txt", "a") as file:
                            file.write(res + "\n")
                    # print("---")
