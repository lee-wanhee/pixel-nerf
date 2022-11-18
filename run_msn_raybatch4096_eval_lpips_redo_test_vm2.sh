#!/bin/bash
python eval/eval.py -n msn_train_raybatch4096_1117 -c conf/exp/msn.conf \
    -D /data/multishapenet --gpu_id='0 1 2 3 4 5 6 7 8 9 10 11 12 13 14' \
    -P '0' -O eval_out/msn_test_1118 \
    --frame5 \
    --n_scenes 15000 --skip 0 --fixed_locality --n_img_each_scene 3 \
    --debug_vis_path checkpoints_msn_test_1118 --no_shuffle \
    --checkpoints_path checkpoints --msn --msn_test_mode 'test' --include_src
    #  -V 1 --batch_size 8
    # --fg_mask
    # --n_scenes 6229 --skip 400