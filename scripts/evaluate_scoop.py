import os
import sys
import torch
import argparse
import numpy as np
import time
from tqdm import tqdm

# add path
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_dir not in sys.path:
    sys.path.append(project_dir)

from models.scoop import SCOOP
from models.refiner import Refiner
from datasets.generic import Batch
from tools.seed import seed_everything
from tools.utils import log_string, iterate_in_chunks
from torch.utils.data import DataLoader


def compute_flow(scoop, batch, args):
    pc_0, pc_1 = batch["sequence"][0], batch["sequence"][1]

    n0 = int(batch["orig_size"][0].cpu().numpy())
    n1 = int(batch["orig_size"][1].cpu().numpy())

    with torch.no_grad():
        est_flow = torch.zeros([1, n0, 3], dtype=torch.float32, device=pc_0.device)
        corr_conf = torch.zeros([1, n0], dtype=torch.float32, device=pc_0.device)

        feats_0, graph = scoop.get_features(pc_0)
        feats_1, _ = scoop.get_features(pc_1)

        b, nb_points0, c = feats_0.shape
        b, nb_points1, c = feats_1.shape

        pc_0_orig = torch.unsqueeze(torch.reshape(pc_0, (b * nb_points0, 3))[:n0], dim=0)
        pc_1_orig = torch.unsqueeze(torch.reshape(pc_1, (b * nb_points1, 3))[:n1], dim=0)
        feats_0_orig = torch.unsqueeze(torch.reshape(feats_0, (b * nb_points0, c))[:n0], dim=0)
        feats_1_orig = torch.unsqueeze(torch.reshape(feats_1, (b * nb_points1, c))[:n1], dim=0)
        idx = np.arange(n0)
        for b in iterate_in_chunks(idx, args.nb_points_chunk):
            points = pc_0_orig[:, b]
            feats = feats_0_orig[:, b]
            points_flow, points_conf, _ = scoop.get_recon_flow([points, pc_1_orig], [feats, feats_1_orig])
            est_flow[:, b] = points_flow
            corr_conf[:, b] = points_conf

    return est_flow, corr_conf, graph


def compute_epe_test(est_flow, batch, args):
    """
    Compute EPE, accuracy and number of outliers.

    Parameters
    ----------
    est_flow : torch.Tensor
        Estimated flow.
    batch : scoop.datasets.generic.Batch
        Contains ground truth flow and mask.
    args : Namespace
        Arguments for evaluation.

    Returns
    -------
    EPE3D : float
        End point error.
    acc3d_strict : float
        Strict accuracy.
    acc3d_relax : float
        Relax accuracy.
    outlier : float
        Percentage of outliers.

    """

    # Extract occlusion mask
    mask = batch["ground_truth"][0].cpu().numpy()[..., 0]

    # Flow
    sf_gt_before_mask = batch["ground_truth"][1].cpu().numpy()
    sf_pred_before_mask = est_flow.cpu().numpy()

    # In all_points evaluation mode, take only the original points of the source point cloud
    n1 = batch["orig_size"][0].cpu().numpy()
    if args.all_points:
        assert len(n1) == 1, "If evaluating with all points, the batch size should be equal 1 (got %d)" % len(n1)

        mask = mask.reshape(-1)[:int(n1)]

        sf_gt_before_mask = sf_gt_before_mask.reshape([-1, 3])[:int(n1)]
        sf_pred_before_mask = sf_pred_before_mask.reshape([-1, 3])[:int(n1)]

    # Flow
    sf_gt = sf_gt_before_mask[mask > 0]
    sf_pred = sf_pred_before_mask[mask > 0]

    # EPE
    epe3d_per_point = np.linalg.norm(sf_gt - sf_pred, axis=-1)
    epe3d = epe3d_per_point.mean()

    #
    sf_norm = np.linalg.norm(sf_gt, axis=-1)
    relative_err_per_point = epe3d_per_point / (sf_norm + 1e-4)
    acc3d_strict_per_point = (np.logical_or(epe3d_per_point < 0.05, relative_err_per_point < 0.05)).astype(np.float32)
    acc3d_strict = acc3d_strict_per_point.mean()
    acc3d_relax_per_point = (np.logical_or(epe3d_per_point < 0.1, relative_err_per_point < 0.1)).astype(np.float32)
    acc3d_relax = acc3d_relax_per_point.mean()
    outlier_per_point = (np.logical_or(epe3d_per_point > 0.3, relative_err_per_point > 0.1)).astype(np.float32)
    outlier = outlier_per_point.mean()

    return epe3d, acc3d_strict, acc3d_relax, outlier, epe3d_per_point, acc3d_strict_per_point, acc3d_relax_per_point, outlier_per_point


def eval_model(scoop, testloader, log_file, log_dir, res_dir, args):
    """
    Compute performance metrics on test / validation set.

    Parameters
    ----------
    scoop : scoop.models.SCOOP
        SCOOP model to evaluate.
    testloader : scoop.datasets.generic.SceneFlowDataset
        Dataset  loader.
    log_file: file
        Evaluation log file.
    log_dir:
        Directory for saving results.
    res_dir: srt
        Directory for saving point cloud results (active if not None).
    args : Namespace
        Arguments for evaluation.

    Returns
    -------
    mean_epe : float
        Average EPE on dataset.
    mean_outlier : float
        Average percentage of outliers.
    mean_acc3d_relax : float
        Average relaxed accuracy.
    mean_acc3d_strict : TYPE
        Average strict accuracy.

    """

    # Init.
    start_time_eval = time.time()
    num_batch = len(testloader)
    fname_list = [None] * num_batch
    epe_per_scene = np.zeros(num_batch, dtype=np.float32)
    acc3d_strict_per_scene = np.zeros(num_batch, dtype=np.float32)
    acc3d_relax_per_scene = np.zeros(num_batch, dtype=np.float32)
    outlier_per_scene = np.zeros(num_batch, dtype=np.float32)
    duration_per_scene = np.zeros(num_batch, dtype=np.float32)
    epe_per_point_list = [None] * num_batch
    acc3d_strict_per_point_list = [None] * num_batch
    acc3d_relax_per_point_list = [None] * num_batch
    outlier_per_point_list = [None] * num_batch
    running_epe = 0
    running_acc3d_strict = 0
    running_acc3d_relax = 0
    running_outlier = 0

    if args.use_test_time_refinement:
        target_recon_loss_refinement = np.zeros([num_batch, args.test_time_num_step + 1], dtype=np.float32)
        smooth_flow_loss_refinement = np.zeros([num_batch, args.test_time_num_step + 1], dtype=np.float32)
        epe_refinement = np.zeros([num_batch, args.test_time_num_step + 1], dtype=np.float32)
        duration_refinement = np.zeros([num_batch], dtype=np.float32)

    save_pc_res = res_dir is not None

    #
    scoop = scoop.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for it, batch in enumerate(tqdm(testloader)):

        # Current file name
        fname = os.path.split(testloader.dataset.filename_curr)[-1]
        fname_list[it] = fname

        # Send data to GPU
        batch = batch.to(device, non_blocking=True)

        # Estimate flow
        start_time = time.time()
        if args.all_candidates:
            est_flow, corr_conf, graph = compute_flow(scoop, batch, args)
        else:
            with torch.no_grad():
                est_flow, corr_conf, _, graph = scoop(batch["sequence"])
        duration_corr = time.time() - start_time

        duration_total = duration_corr
        if args.use_test_time_refinement:
            refiner = Refiner(est_flow.shape, est_flow.device)
            test_time_optimizer = torch.optim.Adam(refiner.parameters(), lr=args.test_time_update_rate)
            refined_flow, refine_metrics_curr, duration_curr = refiner.refine_flow(batch, est_flow.detach(), corr_conf.detach(), test_time_optimizer, args)
            est_flow = refined_flow.detach()

            target_recon_loss_refinement[it] = refine_metrics_curr["target_recon_loss_all"]
            smooth_flow_loss_refinement[it] = refine_metrics_curr["smooth_flow_loss_all"]
            epe_refinement[it] = refine_metrics_curr["epe_all"]
            duration_refinement[it] = duration_curr

            duration_total = duration_total + duration_curr

        # Performance metrics
        epe3d, acc3d_strict, acc3d_relax, outlier, epe3d_pp, acc3d_strict_pp, acc3d_relax_pp, outlier_pp =\
            compute_epe_test(est_flow, batch, args)
        epe_per_scene[it] = epe3d
        acc3d_strict_per_scene[it] = acc3d_strict
        acc3d_relax_per_scene[it] = acc3d_relax
        outlier_per_scene[it] = outlier
        duration_per_scene[it] = duration_total
        epe_per_point_list[it] = epe3d_pp
        acc3d_strict_per_point_list[it] = acc3d_strict_pp
        acc3d_relax_per_point_list[it] = acc3d_relax_pp
        outlier_per_point_list[it] = outlier_pp
        running_epe += epe3d
        running_outlier += outlier
        running_acc3d_relax += acc3d_relax
        running_acc3d_strict += acc3d_strict

        # Save point cloud results
        if save_pc_res:
            n1 = int(batch["orig_size"][0])
            n1_save = n1 if args.all_points else int(batch["sequence"][0].shape[1])
            pc1_save = batch["sequence"][0].cpu().numpy().reshape([-1, 3])[:n1_save]
            est_flow_save = est_flow.cpu().numpy().reshape([-1, 3])[:n1_save]
            corr_conf_save = corr_conf.cpu().numpy().reshape([-1])[:n1_save]
            mask_save = batch["ground_truth"][0].cpu().numpy().reshape([-1, 1])[:n1_save] == 1
            gt_save = batch["ground_truth"][1].cpu().numpy().reshape([-1, 3])[:n1_save]

            n2 = int(batch["orig_size"][1])
            n2_save = n2 if args.all_points else int(batch["sequence"][1].shape[1])
            pc2_save = batch["sequence"][1].cpu().numpy().reshape([-1, 3])[:n2_save]

            fname_split = fname.split(".")
            fname_pc_save = ".".join([fname_split[0] + "_res", fname_split[1]])
            path_pc_save = os.path.join(res_dir, fname_pc_save)
            np.savez(path_pc_save, pc1=pc1_save, pc2=pc2_save, gt_mask_for_pc1=mask_save, gt_flow_for_pc1=gt_save, est_flow_for_pc1=est_flow_save, corr_conf_for_pc1=corr_conf_save)

    mean_epe = running_epe / num_batch
    mean_outlier = running_outlier / num_batch
    mean_acc3d_relax = running_acc3d_relax / num_batch
    mean_acc3d_strict = running_acc3d_strict / num_batch

    log_string(log_file, "EPE: %.4f, ACC3DS: %.4f, ACC3DR: %.4f, Outlier: %.4f, Dataset Size: %d" %
               (mean_epe, mean_acc3d_strict, mean_acc3d_relax, mean_outlier, num_batch))

    duration_eval = time.time() - start_time_eval
    log_string(log_file, "Evaluation duration: %.2f minutes (time per example: %.2f seconds)" %
               (duration_eval/60, duration_eval/num_batch))

    fnames = np.array(fname_list)
    if args.save_metrics:
        path_metrics_save = os.path.join(log_dir, args.metrics_fname)
        data_for_save = {"fnames": fnames,
                         "epe_per_scene": epe_per_scene,
                         "acc3d_strict_per_scene": acc3d_strict_per_scene,
                         "acc3d_relax_per_scene": acc3d_relax_per_scene,
                         "outlier_per_scene": outlier_per_scene,
                         "duration_per_scene": duration_per_scene,
                         "epe_per_point": epe_per_point_list
                         }

        if args.use_test_time_refinement:
            data_for_save["target_recon_loss_refinement"] = target_recon_loss_refinement
            data_for_save["smooth_flow_loss_refinement"] = smooth_flow_loss_refinement
            data_for_save["epe_refinement"] = epe_refinement
            data_for_save["duration_refinement"] = duration_refinement

        np.savez(path_metrics_save, **data_for_save)

    return mean_epe, mean_outlier, mean_acc3d_relax, mean_acc3d_strict


def my_main(args):
    """
    Entry point of the script.

    Parameters
    ----------
    args.dataset_name : str
        Dataset for evaluation. Either FlowNet3D_kitti or FlowNet3D_ft3d or HPLFlowNet_kitti or HPLFlowNet_ft3d.
    args.batch_size: int
        Batch size for evaluation.
    args.nb_points : int
        Number of points in point clouds.
    args.path2ckpt : str
        Path to saved model.
    args.mode : str
        Whether to use test set of validation set or all set.

    Raises
    ------
    ValueError
        Unknown dataset.

    """

    # Set seed
    seed = seed_everything(seed=42)

    # Path to current file
    pathroot = os.path.dirname(__file__)

    # Select dataset
    if args.dataset_name.split("_")[0].lower() == "HPLFlowNet".lower():

        # HPLFlowNet version of the datasets
        path2data = os.path.join(pathroot, "..", "data", "HPLFlowNet")

        # KITTI
        if args.dataset_name.split("_")[1].lower() == "kitti".lower():
            path2data = os.path.join(path2data, "KITTI_processed_occ_final")
            from datasets.kitti_hplflownet import Kitti

            assert args.mode == "val" or args.mode == "test" or args.mode == "all", "Problem with mode " + args.mode
            dataset = Kitti(root_dir=path2data, nb_points=args.nb_points, all_points=args.all_points, mode=args.mode)

        # FlyingThing3D
        elif args.dataset_name.split("_")[1].lower() == "ft3d".lower():
            path2data = os.path.join(path2data, "FlyingThings3D_subset_processed_35m")
            from datasets.flyingthings3d_hplflownet import FT3D

            assert args.mode == "val" or args.mode == "test", "Problem with mode " + args.mode
            dataset = FT3D(root_dir=path2data, nb_points=args.nb_points, all_points=args.all_points, mode=args.mode, nb_examples=-1)

        else:
            raise ValueError("Unknown dataset " + args.dataset_name)

    elif args.dataset_name.split("_")[0].lower() == "FlowNet3D".lower():

        # FlowNet3D version of the datasets
        path2data = os.path.join(pathroot, "..", "data", "FlowNet3D")

        # KITTI
        if args.dataset_name.split("_")[1].lower() == "kitti".lower():
            path2data = os.path.join(path2data, "kitti_rm_ground")
            from datasets.kitti_flownet3d import Kitti

            assert args.mode == "val" or args.mode == "test" or args.mode == "all", "Problem with mode " + args.mode
            dataset = Kitti(root_dir=path2data, nb_points=args.nb_points, all_points=args.all_points,
                            same_v_t_split=True, mode=args.mode)

        # FlyingThing3D
        elif args.dataset_name.split("_")[1].lower() == "ft3d".lower():
            path2data = os.path.join(path2data, "data_processed_maxcut_35_20k_2k_8192")
            from datasets.flyingthings3d_flownet3d import FT3D

            assert args.mode == "val" or args.mode == "test", "Problem with mode " + args.mode
            dataset = FT3D(root_dir=path2data, nb_points=args.nb_points, all_points=args.all_points, mode=args.mode, nb_examples=-1)

        else:
            raise ValueError("Unknown dataset" + args.dataset_name)

    else:
        raise ValueError("Unknown dataset " + args.dataset_name)
    print("\n\nDataset: " + path2data + " " + args.mode)

    # Dataloader
    testloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=args.nb_workers,
        collate_fn=Batch,
        drop_last=False,
    )

    # Load Checkpoint
    file = torch.load(args.path2ckpt)

    # Load parameters
    saved_args = file["args"]

    # Load model
    scoop = SCOOP(saved_args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    scoop = scoop.to(device, non_blocking=True)
    scoop.load_state_dict(file["model"])
    scoop = scoop.eval()

    # Log file
    log_dir = os.path.split(args.path2ckpt)[0]
    log_file = open(os.path.join(log_dir, args.log_fname), 'w')
    log_string(log_file, 'Evaluation arguments:')
    log_file.write(str(args) + '\n')
    log_string(log_file, "Seed: %d" % seed)

    # Point cloud results directory
    res_dir = None
    if args.save_pc_res:
        res_dir = os.path.join(log_dir, args.res_folder)
        if not os.path.exists(res_dir):
            os.mkdir(res_dir)

    # Evaluation
    epsilon = 0.03 + torch.exp(scoop.epsilon).item()
    gamma = torch.exp(scoop.gamma).item()
    power = gamma / (gamma + epsilon)
    log_string(log_file, "Epsilon: %.4f, Power: %.4f" % (epsilon, power))
    eval_model(scoop, testloader, log_file, log_dir, res_dir, args)

    log_string(log_file, "Finished Evaluation.")
    log_file.close()


if __name__ == "__main__":

    # Args
    parser = argparse.ArgumentParser(description="Evaluate SCOOP.")
    parser.add_argument("--dataset_name", type=str, default="FlowNet3D_kitti", help="Dataset. FlowNet3D_kitti or FlowNet3D_FT3D or Either HPLFlowNet_kitti or HPLFlowNet_FT3D.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation.")
    parser.add_argument("--mode", type=str, default="test", help="Test or validation or all dataset (options: [val, test, all]).")
    parser.add_argument("--use_test_time_refinement", type=int, default=1, help="1: Use test time refinement, 0: Do not use test time refinement.")
    parser.add_argument("--test_time_num_step", type=int, default=150, help="1: Number of steps for test time refinement.")
    parser.add_argument("--test_time_update_rate", type=float, default=0.2, help="1: Update rate for test time refinement.")
    parser.add_argument("--backward_dist_weight", type=float, default=0.0, help="Backward distance weight for target reconstruction loss in test time refinement.")
    parser.add_argument("--target_recon_loss_weight", type=float, default=1.0, help="Weight for target reconstruction loss in test time refinement.")
    parser.add_argument("--use_smooth_flow", type=int, default=1, help="1: Use self smooth flow loss in test time refinement, 0: Do not use smooth flow loss.")
    parser.add_argument("--nb_neigh_smooth_flow", type=int, default=32, help="Number of neighbor points for smooth flow loss in test time refinement.")
    parser.add_argument("--smooth_flow_loss_weight", type=float, default=1.0, help="Weight for smooth flow loss in test time refinement. Active if > 0.")
    parser.add_argument("--test_time_verbose", type=int, default=0, help="1: Print test time results during optimization, 0: Do not print.")
    parser.add_argument("--use_chamfer_cuda", type=int, default=1, help="1: Use chamfer distance cuda implementation in test time refinement, 0: Use chamfer distance pytorch implementation in test time refinement.")
    parser.add_argument("--nb_points", type=int, default=2048, help="Maximum number of points in point cloud.")
    parser.add_argument("--all_points", type=int, default=1, help="1: use all point in the source point cloud for evaluation in chunks of nb_points, 0: use only nb_points.")
    parser.add_argument("--all_candidates", type=int, default=1, help="1: use all points in the target point cloud as candidates concurrently, 0: use chunks of nb_points from the target point cloud each time.")
    parser.add_argument("--nb_points_chunk", type=int, default=2048, help="Number of source points chuck for evaluation with all candidate target points.")
    parser.add_argument("--nb_workers", type=int, default=0, help="Number of workers for the dataloader.")
    parser.add_argument("--path2ckpt", type=str, default="./../pretrained_models/kitti_v_100_examples/model_e400.tar", help="Path to saved checkpoint.")
    parser.add_argument("--log_fname", type=str, default="log_evaluation.txt", help="Evaluation log file name.")
    parser.add_argument("--save_pc_res", type=int, default=0, help="1: save point cloud results, 0: do not save point cloud results [default: 0]")
    parser.add_argument("--res_folder", type=str, default="pc_res", help="Folder name for saving results.")
    parser.add_argument("--save_metrics", type=int, default=0, help="1: save evaluation metrics results, 0: do not save evaluation metrics results [default: 0]")
    parser.add_argument("--metrics_fname", type=str, default="metrics_results.npz", help="Name for metrics file.")
    args = parser.parse_args()

    # Check arguments
    if args.all_points:
        assert args.batch_size == 1, "For evaluation with all source points, the batch_size should be equal to 1 (got %d)" % args.batch_size

    if args.all_candidates:
        assert args.batch_size == 1, "For evaluation with all candidate target points, the batch_size should be equal to 1 (got %d)" % args.batch_size

    if args.save_pc_res:
        assert args.batch_size == 1, "For evaluation with saving point cloud results, the batch_size should be equal to 1 (got %d)" % args.batch_size

    # Launch evaluation
    my_main(args)
