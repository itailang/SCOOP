#!/usr/bin/env bash

# train on ft3d_s
python train_scoop.py --dataset_name HPLFlowNet_ft3d --nb_train_examples 1800 --nb_val_examples 200 --nb_points 8192 \
    --batch_size_train 1 --batch_size_val 1 --nb_epochs 60 --nb_workers 8 \
    --backward_dist_weight 1.0 --use_corr_conf 0 \
    --add_model_suff 1 --save_model_epoch 15 --log_dir ft3d_s_1800_examples
