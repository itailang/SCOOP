""" Loss functions. """
import os
import sys
import torch
import numpy as np

# add path
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_dir not in sys.path:
    sys.path.append(project_dir)

try:
    from auxiliary.ChamferDistancePytorch.chamfer3D import dist_chamfer_3D
    chamfer_dist_3d_cu = dist_chamfer_3D.chamfer_3DDist()
except:
    print("Could not load compiled 3D CUDA chamfer distance")

from tools.utils import iterate_in_chunks


def compute_loss_unsupervised(recon_flow, corr_conf, target_pc_recon, graph, batch, args):
    """
    Compute unsupervised training loss.

    Parameters
    ----------
    recon_flow: torch.Tensor
        Flow from reconstruction of the target point cloud by the source point cloud.
    corr_conf: torch.Tensor
        Correspondence confidence.
    target_pc_recon: torch.Tensor
        Cross reconstructed target point cloud.
    graph: scoop.models.graph.Graph
        Nearest neighbor graph for the source point cloud.
    batch: scoop.datasets.generic.Batch
        Contains ground truth flow and mask.
    args: dictionary.
        Arguments for loss terms.

    Returns
    -------
    loss : torch.Tensor
        Training loss for current batch.

    """
    mask = None

    if args.use_corr_conf:
        point_weight = corr_conf
    else:
        point_weight = None

    target_pc_input = batch["sequence"][1]
    target_recon_loss = chamfer_loss(target_pc_recon, target_pc_input, point_weight, args.backward_dist_weight, mask)

    loss = target_recon_loss

    if args.use_corr_conf and args.corr_conf_loss_weight > 0:
        if mask is not None:
            corr_conf_masked = corr_conf[mask > 0]
        else:
            corr_conf_masked = corr_conf

        corr_conf_loss = 1 - torch.mean(corr_conf_masked)
        loss = loss + (args.corr_conf_loss_weight * corr_conf_loss)
    else:
        corr_conf_loss = 0

    if args.use_smooth_flow and args.smooth_flow_loss_weight > 0:
        smooth_flow_loss, _ = smooth_loss(recon_flow, graph, args.nb_neigh_smooth_flow, loss_norm=1, mask=mask)
        loss = loss + (args.smooth_flow_loss_weight * smooth_flow_loss)
    else:
        smooth_flow_loss = 0

    return loss, target_recon_loss, corr_conf_loss, smooth_flow_loss


def chamfer_dist_3d_pt(pc1, pc2, backward_dist_weight=0.0, chunk_size=2048):
    """
    Compute Chamfer Distance between two point clouds.
    Input:
        pc1: (b, n, 3) torch.Tensor, first point cloud xyz coordinates.
        pc2: (b, m, 3) torch.Tensor, second point cloud xyz coordinates.
        backward_dist_weight: float, weight for backward distance
        chunk_size: int, chunk size for distance computation.

    Output:
        dist1: (b, n) torch.Tensor float32, for each point in pc1, the distance to the closest point in pc2.
        dist2: (b, m) torch.Tensor float32, for each point in pc2, the distance to the closest point in pc1.
        idx1: (b, n) torch.Tensor int32, for each point in pc1, the index of the closest point in pc2 (values are in the range [0, ..., m-1]).
        idx2: (b, m) torch.Tensor int32, for each point in pc2, the index of the closest point in pc1 (values are in the range [0, ..., n-1]).
    """

    b = pc1.shape[0]
    n = pc1.shape[1]
    m = pc2.shape[1]
    device = pc1.device

    dist1 = torch.zeros([b, n], dtype=torch.float32, device=device)
    idx1 = torch.zeros([b, n], dtype=torch.int32, device=device)

    rng1 = np.arange(n)
    for chunk in iterate_in_chunks(rng1, chunk_size):
        pc1_curr = torch.unsqueeze(pc1[:, chunk], dim=2).repeat(1, 1, m, 1)
        pc2_curr = torch.unsqueeze(pc2, dim=1).repeat(1, len(chunk), 1, 1)
        diff = pc1_curr - pc2_curr  # shape (b, cs, m, 3)
        dist = torch.sum(diff ** 2, dim=-1)  # shape (b, cs, m)

        min1 = torch.min(dist, dim=2)
        dist1_curr = min1.values
        idx1_curr = min1.indices.type(torch.IntTensor)
        idx1_curr = idx1_curr.to(dist.device)

        dist1[:, chunk] = dist1_curr
        idx1[:, chunk] = idx1_curr

    if backward_dist_weight == 0.0:
        dist2 = None
        idx2 = None
    else:
        dist2 = torch.zeros([b, m], dtype=torch.float32, device=device)
        idx2 = torch.zeros([b, m], dtype=torch.int32, device=device)

        rng2 = np.arange(m)
        for chunk in iterate_in_chunks(rng2, chunk_size):
            pc1_curr = torch.unsqueeze(pc1, dim=2).repeat(1, 1, len(chunk), 1)
            pc2_curr = torch.unsqueeze(pc2[:, chunk], dim=1).repeat(1, n, 1, 1)
            diff = pc1_curr - pc2_curr  # shape (b, n, cs, 3)
            dist = torch.sum(diff ** 2, dim=-1)  # shape (b, n, cs)

            min2 = torch.min(dist, dim=1)
            dist2_curr = min2.values
            idx2_curr = min2.indices.type(torch.IntTensor)
            idx2_curr = idx2_curr.to(dist.device)

            dist2[:, chunk] = dist2_curr
            idx2[:, chunk] = idx2_curr

    return dist1, dist2, idx1, idx2


def chamfer_loss(pc1, pc2, point_weight=None, backward_dist_weight=1.0, mask=None, use_chamfer_cuda=True):
    if not pc1.is_cuda:
        pc1 = pc1.cuda()

    if not pc2.is_cuda:
        pc2 = pc2.cuda()

    if use_chamfer_cuda:
        dist1, dist2, idx1, idx2 = chamfer_dist_3d_cu(pc1, pc2)
    else:
        dist1, dist2, idx1, idx2 = chamfer_dist_3d_pt(pc1, pc2, backward_dist_weight)

    if point_weight is not None:
        dist1_weighted = dist1 * point_weight
    else:
        dist1_weighted = dist1

    if mask is not None:
        dist1_masked = dist1_weighted[mask > 0]
        dist1_mean = torch.mean(dist1_masked)
    else:
        dist1_mean = torch.mean(dist1_weighted)

    if backward_dist_weight == 1.0:
        loss = dist1_mean + torch.mean(dist2)
    elif backward_dist_weight == 0.0:
        loss = dist1_mean
    else:
        loss = dist1_mean + backward_dist_weight * torch.mean(dist2)

    return loss


def smooth_loss(est_flow, graph, nb_neigh, loss_norm=1, mask=None):
    b, n, c = est_flow.shape
    est_flow_neigh = est_flow.reshape(b * n, c)
    est_flow_neigh = est_flow_neigh[graph.edges]
    est_flow_neigh = est_flow_neigh.view(b, n, graph.k_neighbors, c)
    est_flow_neigh = est_flow_neigh[:, :, 1:(nb_neigh + 1)]
    flow_diff = (est_flow.unsqueeze(2) - est_flow_neigh).norm(p=loss_norm, dim=-1)

    if mask is not None:
        mask_neigh = mask.reshape(b * n)
        mask_neigh = mask_neigh[graph.edges]
        mask_neigh = mask_neigh.view(b, n, graph.k_neighbors)
        mask_neigh = mask_neigh[:, :, 1:(nb_neigh + 1)]
        mask_neigh_sum = mask_neigh.sum(dim=-1)

        flow_diff_masked = flow_diff * mask_neigh
        flow_diff_masked_sum = flow_diff_masked.sum(dim=-1)
        smooth_flow_per_point = flow_diff_masked_sum / (mask_neigh_sum + 1e-8)
        smooth_flow_per_point = smooth_flow_per_point[mask > 0]
    else:
        smooth_flow_per_point = flow_diff.mean(dim=-1)

    smooth_flow_loss = smooth_flow_per_point.mean()

    return smooth_flow_loss, smooth_flow_per_point
