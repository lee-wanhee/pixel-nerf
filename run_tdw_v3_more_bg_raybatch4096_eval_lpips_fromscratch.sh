#!/bin/bash
python eval/eval.py -n tdw_v3_more_bg_train_raybatch4096_fromscratch -c conf/exp/tdw.conf \
    -D /ccn2/u/honglinc/datasets/tdw_playroom_v3_more_bg_val --gpu_id='0' \
    -P '0' -O eval_out/tdw_v3_more_bg_train_raybatch4096_fromscratch_120723 \
    --frame5 \
    --n_scenes 9039 --skip 0 --fixed_locality --n_img_each_scene 4 \
    --debug_vis_path vis_tdw_v3_more_bg_raybatch4096_fromscratch_120723 --no_shuffle \
    --checkpoints_path checkpoints
    #  -V 1 --batch_size 8
    # --fg_mask
    # --n_scenes 6229 --skip 400



#python train/train.py -n tdw_train_raybatch4096 -c conf/exp/tdw.conf \
#    -D /data/tdw_playroom_v2_train_combine -V 1 --gpu_id='0 1 2 3' --frame5 \
#    --n_scenes 15000 --skip 0 --fixed_locality --n_img_each_scene 4 \
#    --ray_batch_size 4096 --batch_size 8
#    # --fg_mask
    # --n_scenes 6229 --skip 400