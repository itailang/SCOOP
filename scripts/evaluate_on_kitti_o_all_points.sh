#!/usr/bin/env bash

# evaluate on kitti_o with all the points
python evaluate_scoop.py --dataset_name FlowNet3D_kitti --mode all --nb_points 2048 \
    --path2ckpt ./../experiments/ft3d_o_1800_examples/model_e100.tar \
    --use_test_time_refinement 1 --test_time_num_step 150 --test_time_update_rate 0.2 \
    --log_fname log_evaluation_kitti_o_all_points.txt
