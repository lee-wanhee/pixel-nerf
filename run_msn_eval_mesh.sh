#!/bin/bash
z_limit="${1:-$'0.0'}"
isosurface="${2:-$'10.0'}"
radius="${3:-$'10.0'}"
echo "z_limit = $z_limit"
echo "isosurface = $isosurface"
echo "radius = $radius"
python eval/eval_mesh.py -n msn_train_raybatch4096_1117 -c conf/exp/msn.conf \
    -D /ccn2/u/honglinc/multishapenet --gpu_id='0' \
    -P '0' -O eval_out/msn_mesh_030723_11pm \
    --frame5 \
    --n_scenes 15000 --skip 8000 --fixed_locality --n_img_each_scene 3 --no_shuffle \
    --resume --checkpoints_path checkpoints_backup_030723 \
    --z_limit $z_limit --isosurface $isosurface --radius $radius \
    --use_eisen_seg --unmasked_mesh --msn --msn_test_mode 'val' \

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