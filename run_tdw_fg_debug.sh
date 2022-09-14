#!/bin/bash
python train/train.py -n tdw_train_fg -c conf/exp/tdw_fg.conf \
    -D /ccn2/u/honglinc/datasets/tdw_playroom_v2_train_combine -V 1 --gpu_id='0' --frame5 \
    --n_scenes 100 --skip 0 --fixed_locality --n_img_each_scene 4 \
    --ray_batch_size 128 --batch_size 1 --fg_mask
    # --n_scenes 6229 --skip 400