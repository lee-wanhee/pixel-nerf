#!/bin/bash
python eval/eval.py -n msn_train_raybatch4096_111_backup_3am -c conf/exp/msn.conf \
    -D /ccn2/u/honglinc/datasets/multishapenet --gpu_id='0 1 2 3' \
    -P '0' -O eval_out/msn_val_1117 \
    --frame5 \
    --n_scenes 15000 --skip 0 --fixed_locality --n_img_each_scene 3 \
    --debug_vis_path checkpoints_msn_val_1118_3am --no_shuffle \
    --checkpoints_path checkpoints --msn --msn_test_mode 'val' --include_src
    #  -V 1 --batch_size 8
    # --fg_mask
    # --n_scenes 6229 --skip 400