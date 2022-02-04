#!/bin/bash

python train.py --model_def ./config/yolo-nano.cfg --data_config config/coco.data --batch_size 8 --epochs 10
