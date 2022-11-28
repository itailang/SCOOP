""" Helper utility functions. """
import os
import numpy as np


def create_dir(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
    return directory


def log_string(log_file, log_str):
    log_file.write(log_str + '\n')
    log_file.flush()
    print(log_str)


def iterate_in_chunks(l, n):
    """ Yield successive 'n'-sized chunks from iterable 'l'.
    Note: last chunk will be smaller than l if n doesn't divide l perfectly.
    """
    for i in range(0, len(l), n):
        yield l[i: i + n]


def load_data(load_path):
    pc_res = np.load(load_path)

    pc1, pc2, gt_flow, est_flow = pc_res['pc1'], pc_res['pc2'], pc_res['gt_flow_for_pc1'], pc_res['est_flow_for_pc1']
    try:
        gt_mask = pc_res['gt_mask_for_pc1']
        gt_mask = gt_mask.reshape(len(pc1))
    except KeyError:
        gt_mask = np.ones(len(pc1)) == 1

    try:
        corr_conf = pc_res['corr_conf_for_pc1']
    except KeyError:
        corr_conf = np.ones(len(pc1), dtype=np.float32)

    pc1_warped_gt = pc1 + gt_flow
    pc1_warped_est = pc1 + est_flow

    return pc1, pc2, gt_mask, gt_flow, est_flow, corr_conf, pc1_warped_gt, pc1_warped_est

