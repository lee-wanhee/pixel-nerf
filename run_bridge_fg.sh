#!/bin/bash
python train/train.py -n bridge_train_fg -c conf/exp/tdw_fg.conf \
    -D /data/bridge_multiview_v1_train -V 1 --gpu_id='0 1 2 3' --frame5 \
    --n_scenes 6230 --skip 0 --fixed_locality --n_img_each_scene 3 \
    --ray_batch_size 4096 --batch_size 8 --small_dataset --fg_mask
    # --n_scenes 6229 --skip 400
    # CUDA 8,9,10,11