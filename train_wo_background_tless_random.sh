#!/bin/bash

# Assuming you have a bash script named 'train_pix2pose.sh'

for i in {10..30}; do
  python3 tools/3_train_pix2pose_wo_background_crop.py 0 cfg/cfg_bop2020_rgb_custom.json tless_random_texture $i
done