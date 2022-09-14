#!/bin/bash
python train/train.py -n bridge_train_2 -c conf/exp/tdw.conf \
    -D /ccn2/u/honglinc/datasets/bridge_multiview_v1_train -V 1 --gpu_id='0' --frame5 \
    --n_scenes 6229 --skip 0 --fixed_locality --n_img_each_scene 3 --small_dataset \
    --ray_batch_size 4096 --batch_size 1
    # --n_scenes 6229 --skip 400