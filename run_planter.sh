#!/bin/bash
python train/train.py -n planter_raybatch4096 -c conf/exp/planter.conf \
    -D /ccn2/u/wanhee/datasets/planters/4obj-train -V 1 --gpu_id='0 1 2 3 4 5 6 7' --frame5 \
    --n_scenes 745 --skip 0 --fixed_locality --n_img_each_scene 3 \
    --ray_batch_size 4096 --batch_size 8 \