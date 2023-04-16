#!/usr/bin/env bash

# evaluate on kitti_o
python evaluate_scoop.py --dataset_name FlowNet3D_kitti --mode all --nb_points 2048 --all_points 0 --all_candidates 0 \
    --path2ckpt ./../experiments/ft3d_o_1800_examples/model_e100.tar \
    --use_test_time_refinement 1 --test_time_num_step 1000 --test_time_update_rate 0.05 \
    --log_fname log_evaluation_kitti_o.txt
