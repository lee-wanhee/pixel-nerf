#!/bin/bash
python train/train.py -n tdw_v3_more_bg_train_raybatch4096_fromscratch -c conf/exp/tdw.conf \
    -D /ccn2/u/honglinc/datasets/tdw_playroom_v3_more_bg -V 1 --gpu_id='0 1 2 3 4 5 6 7' --frame5 \
    --n_scenes 9039 --skip 0 --fixed_locality --n_img_each_scene 4 \
    --ray_batch_size 4096 --batch_size 8 \
