#!/bin/bash
python eval/eval_mesh.py -n tdw_train_raybatch4096 -c conf/exp/tdw.conf \
    -D /ccn2/u/honglinc/datasets/tdw_playroom_v2_test_combine --gpu_id='0' \
    -P '0' -O eval_out/tdw_test_mesh_43_10 \
    --frame5 \
    --n_scenes 15000 --skip 0 --fixed_locality --n_img_each_scene 4 --no_shuffle \
    --resume --checkpoints_path checkpoints \
    --z_limit 4.3 --isosurface 10 \

#    --debug_vis_path tdw_val_mesh_43_30_vis_path
#    --ray_batch_size 4096
    #  -V 1 --batch_size 8
    # --fg_mask
    # --n_scenes 6229 --skip 400



#python train/train.py -n tdw_train_raybatch4096 -c conf/exp/tdw.conf \
#    -D /data/tdw_playroom_v2_train_combine -V 1 --gpu_id='0 1 2 3' --frame5 \
#    --n_scenes 15000 --skip 0 --fixed_locality --n_img_each_scene 4 \
#    --ray_batch_size 4096 --batch_size 8
#    # --fg_mask
    # --n_scenes 6229 --skip 400