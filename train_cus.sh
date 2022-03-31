#!/bin/bash

#CUDA_LAUNCH_BLOCKING=1 python train.py --model_def ./config/yolo-nano_cus.cfg --data_config config/custom.data --batch_size 2 --epochs 10
#python train.py --model_def ./config/yolo-nano_cus.cfg --data_config config/custom.data --batch_size 8 --epochs 200
CUDA_VISIBLE_DEVICES=0 python train.py --model_def ./config/yolo-nano_cus.cfg --data_config config/custom.data --batch_size 8 --epochs 200

