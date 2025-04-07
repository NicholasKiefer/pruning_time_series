#!/bin/bash

cd ..
CUDA_VISIBLE_DEVICES=2 python main_crossformer.py --data ETTm2 --in_len 672 --out_len 96 --seg_len 12 --learning_rate 1e-4 --itr 3 &
CUDA_VISIBLE_DEVICES=3 python main_crossformer.py --data ETTm2 --in_len 672 --out_len 192 --seg_len 24 --learning_rate 1e-5 --itr 3 &
CUDA_VISIBLE_DEVICES=1 python main_crossformer.py --data ETTm2 --in_len 672 --out_len 336 --seg_len 24 --learning_rate 1e-5 --itr 3 &
CUDA_VISIBLE_DEVICES=0 python main_crossformer.py --data ETTm2 --in_len 672 --out_len 720 --seg_len 24 --learning_rate 1e-5 --itr 3 &
wait