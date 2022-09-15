#!/bin/bash
python train/train.py -n bridge_train_largebatch -c conf/exp/tdw.conf \
    -D /data/bridge_multiview_v1_train -V 1 --gpu_id='0 1 2 3' --frame5 \
    --n_scenes 6230 --skip 0 --fixed_locality --n_img_each_scene 3 \
    --ray_batch_size 128 --batch_size 128 --small_dataset
    # --n_scenes 6229 --skip 400
    # 0 1 2 3