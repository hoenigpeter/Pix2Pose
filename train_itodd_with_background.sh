#!/bin/bash

# Assuming you have a bash script named 'train_pix2pose.sh'

for i in {1..28}; do
  DISPLAY=:0 python3 tools/3_train_pix2pose.py 0 cfg/cfg_bop2020_rgb_custom.json itodd $i 0.2 /Pix2Pose/pix2pose_datasets/coco128/images/train2017
done