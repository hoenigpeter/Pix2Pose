#!/bin/bash

# Assuming you have a bash script named 'train_pix2pose.sh'

for i in {1..30}; do
  DISPLAY=:0 python3 tools/5_evaluation_bop_from_file_single_obj.py 0 cfg/cfg_bop2020_rgb_det_file.json tless_random_texture $i
done