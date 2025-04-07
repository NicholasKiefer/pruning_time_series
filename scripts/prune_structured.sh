#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
source ../venv/bin/activate

/bin/bash ../FEDformer/scripts/run.sh
python ../pruning/tp_prune_fed.py

/bin/bash ../Crossformer/scripts/run.sh
python ../pruning/tp_prune_cross.py
