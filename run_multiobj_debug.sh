#!/bin/bash
python train/train.py -n multi_obj -c conf/exp/multi_obj.conf \
 -D './pixel_nerf_data/multi_chair/' --gpu_id='0'
    # --n_scenes 6229 --skip 400