#!/bin/bash

# Assuming you have a bash script named 'train_pix2pose.sh'

for i in {1..30}; do
  DISPLAY=:0 python3 tools/3_train_pix2pose_wo_background_crop.py 0 cfg/cfg_bop2020_rgb_custom.json mp6d $i 1.0
done