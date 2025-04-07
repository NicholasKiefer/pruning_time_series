#!/bin/bash
cd ..

# CUDA_VISIBLE_DEVICES=0 python main_crossformer.py --data ETTh2 --in_len 168 --out_len 24 --seg_len 6 --learning_rate 1e-4 --itr 3 &
CUDA_VISIBLE_DEVICES=1 python main_crossformer.py --data ETTh2 --in_len 168 --out_len 96 --seg_len 6 --learning_rate 1e-4 --itr 3 &
CUDA_VISIBLE_DEVICES=2 python main_crossformer.py --data ETTh2  --in_len 720 --out_len 192 --seg_len 24 --learning_rate 1e-5 --itr 3 &
CUDA_VISIBLE_DEVICES=3 python main_crossformer.py --data ETTh2 --in_len 720 --out_len 336 --seg_len 24 --learning_rate 1e-5 --itr 3 &
CUDA_VISIBLE_DEVICES=0 python main_crossformer.py --data ETTh2 --in_len 720 --out_len 720 --seg_len 24 --learning_rate 1e-5 --itr 3 &
wait