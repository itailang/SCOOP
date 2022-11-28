#!/usr/bin/env bash

# train on ft3d_o
python train_scoop.py --dataset_name FlowNet3D_ft3d --nb_train_examples 1800 --nb_points 2048 \
    --batch_size_train 4 --batch_size_val 10 --nb_epochs 100 --nb_workers 8 \
    --backward_dist_weight 0.0 --use_corr_conf 1 --corr_conf_loss_weight 0.1 \
    --add_model_suff 1 --save_model_epoch 25 --log_dir ft3d_o_1800_examples
