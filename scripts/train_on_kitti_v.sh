#!/usr/bin/env bash

# train on ft3d_o
python train_scoop.py --dataset_name FlowNet3D_kitti --nb_points 2048 \
    --batch_size_train 4 --batch_size_val 10 --nb_epochs 400 --nb_workers 8 \
    --backward_dist_weight 0.0 --use_corr_conf 1 --corr_conf_loss_weight 0.1 \
    --add_model_suff 1 --save_model_epoch 100 --log_dir kitti_v_100_examples
