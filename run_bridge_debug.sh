#!/bin/bash
python train/train.py -n bridge_100scenes -c conf/exp/tdw.conf \
    -D /data/bridge_multiview_v1_train -V 1 --gpu_id='0' --frame5 \
    --n_scenes 100 --skip 0 --fixed_locality --n_img_each_scene 3 \
    --ray_batch_size 4096 --batch_size 2 --small_dataset \
#    --lr 0.00001
    # --n_scenes 6229 --skip 400
    # 0 1 2 3

