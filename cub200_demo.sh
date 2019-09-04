#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1

python demo.py --data bird --model model/bird-77.22.pth
