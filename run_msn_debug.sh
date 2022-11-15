#!/bin/bash
python train/train.py -n msn_train_raybatch4096 -c conf/exp/msn.conf \
    -D /ccn2/u/honglinc/datasets/multishapenet -V 1 --gpu_id='0' \
    --ray_batch_size 4096 --batch_size 2
    # --fg_mask
    # --n_scenes 6229 --skip 400