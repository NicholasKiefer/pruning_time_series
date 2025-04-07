#!/bin/bash

cd ..

CUDA_VISIBLE_DEVICES=0 python main_crossformer.py --data Exchange --in_len 336 --out_len 96 --seg_len 6 --d_model 64 --d_ff 128 --n_heads 2 --learning_rate 5e-4  --lradj fixed --itr 3 &
CUDA_VISIBLE_DEVICES=1 python main_crossformer.py --data Exchange --in_len 336 --out_len 192 --seg_len 12 --d_model 64 --d_ff 128 --n_heads 2 --learning_rate 5e-4  --lradj fixed --itr 3 & 
CUDA_VISIBLE_DEVICES=2 python main_crossformer.py --data Exchange --in_len 336 --out_len 336 --seg_len 24 --d_model 64 --d_ff 128 --n_heads 2 --learning_rate 5e-4  --lradj fixed --itr 3 & 
CUDA_VISIBLE_DEVICES=3 python main_crossformer.py --data Exchange --in_len 720 --out_len 720 --seg_len 24 --d_model 64 --d_ff 128 --n_heads 2 --learning_rate 5e-5  --lradj fixed  --itr 3 &
wait

# CUDA_VISIBLE_DEVICES=0 python main_crossformer.py --data ECL --in_len 720 --out_len 960 --seg_len 24 --d_model 64 --d_ff 128 --n_heads 2 --learning_rate 5e-5  --lradj fixed  --itr 3 
