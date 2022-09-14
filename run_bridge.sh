#!/bin/bash
python train/train.py -n bridge_train_6229 -c conf/exp/tdw.conf \
    -D /ccn2/u/honglinc/datasets/bridge_multiview_v1_train -V 1 --gpu_id='0' --frame5 \
    --n_scenes 6229 --skip 0 --fixed_locality --n_img_each_scene 3 --no_shuffle --small_dataset \
    # --n_scenes 6229 --skip 400