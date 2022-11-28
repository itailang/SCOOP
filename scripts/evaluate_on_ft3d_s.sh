#!/usr/bin/env bash

# evaluate on ft3d_s
python evaluate_scoop.py --dataset_name HPLFlowNet_ft3d --mode test --nb_points 8192 --all_points 0 --all_candidates 0 \
    --path2ckpt ./../experiments/ft3d_s_1800_examples/model_e060.tar --backward_dist_weight 1.0 \
    --use_test_time_refinement 1 --test_time_num_step 1000 --test_time_update_rate 0.1 --nb_neigh_smooth_flow 16 \
    --log_fname log_evaluation_ft3d_s.txt
