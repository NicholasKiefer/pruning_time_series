# train all models on all datasets all pred lens if not exist in checkpoint dir
# prune them and save result in text file

#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
source ../venv/bin/activate

/bin/bash ../FEDformer/scripts/run.sh
python ../pruning/nn_prune_fed.py

/bin/bash ../Crossformer/scripts/run.sh
python ../pruning/nn_prune_cross.py
