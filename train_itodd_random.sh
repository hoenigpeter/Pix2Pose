#!/bin/bash

# Assuming you have a bash script named 'train_pix2pose.sh'

for i in {15..30}; do
  python3 tools/3_train_pix2pose.py 0 cfg/cfg_bop2020_rgb_custom.json itodd_random_texture $i pix2pose_datasets/coco2017/train2017
done