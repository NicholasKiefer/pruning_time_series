#!/bin/bash

source venv/bin/activate
# export PYTHONPATH="FEDformer:$PYTHONPATH"
# export PYTHONPATH="Crossformer:$PYTHONPATH"
CUDA_VISIBLE_DEVICES=0 python -u finetune.py Crossformer Autoformer FEDformer Informer
