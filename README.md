# Pruning Time Series Transformer

This code implements the experiments for the paper [A Comparative Study of Pruning Methods in Transformer-based Time Series Forecasting](https://arxiv.org/abs/2412.12883).

## Data

Get the data from Informer and Autoformer. 

Informer: [Informer](https://github.com/zhouhaoyi/Informer2020)
Autoformer: [Autoformer](https://github.com/thuml/Autoformer)
## python environment
setup a python environment `venv` in main directory.

## training and pruning
Run the scripts `unsctructured.py` and `structured.py` for the pruning experiments. For the extra fine-tuning step run `finetune.py`.

## Results
will be saved in the respective folders `checkpoints` for the models, and in the `pruning_results.txt` file for the pruning results.
