# Pruning Time Series Transformer

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
