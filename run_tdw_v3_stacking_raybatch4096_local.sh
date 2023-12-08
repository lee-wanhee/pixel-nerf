#!/bin/bash
python train/train.py -n tdw_v3_stacking_train_raybatch4096 -c conf/exp/tdw.conf \
    -D /data2/wanhee/tdw_playroom_v3_stacking -V 1 --gpu_id='0 1 2 3 4 5 6 7' --frame5 \
    --n_scenes 3000 --skip 0 --fixed_locality --n_img_each_scene 4 \
    --ray_batch_size 4096 --batch_size 8 \
    --resume \
