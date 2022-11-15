#!/bin/bash
python train/train.py -n dtu_exp -c conf/exp/dtu.conf -D /data2/wanhee/pixel_nerf_data/rs_dtu_4 -V 3 --gpu_id='0'
