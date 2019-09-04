#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1

python train.py --dataset aircraft --n_classes 100 --base_lr 1e-3 --batch_size 128 --epoch 200 --drop_rate 0.25 --T_k 10 --weight_decay 1e-8 --step 1

sleep 300

python train.py --dataset aircraft --n_classes 100 --base_lr 1e-4 --batch_size 64 --epoch 100 --drop_rate 0.25 --T_k 10 --weight_decay 1e-5 --step 2
