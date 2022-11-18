#!/bin/bash
python eval/eval.py -n msn_train_raybatch4096_1117 -c conf/exp/msn.conf \
    -D /data/multishapenet --gpu_id='0' \
    -P '0' -O eval_out/msn_val_1118 \
    --frame5 \
    --n_scenes 15000 --skip 0 --fixed_locality --n_img_each_scene 3 \
    --debug_vis_path checkpoints_msn_val_1118_debug --no_shuffle \
    --checkpoints_path checkpoints --msn --msn_test_mode 'val' --include_src
    #  -V 1 --batch_size 8
    # --fg_mask
    # --n_scenes 6229 --skip 400