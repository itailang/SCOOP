#!/usr/bin/env bash

# evaluate on kitti_t
python evaluate_scoop.py --dataset_name FlowNet3D_kitti --mode test --nb_points 2048 \
    --path2ckpt ./../experiments/kitti_v_100_examples/model_e400.tar \
    --use_test_time_refinement 1 --test_time_num_step 150 --test_time_update_rate 0.2 \
    --log_fname log_evaluation_kitti_t.txt