import torch
import torch.nn.functional as F


def get_support_matrix(pcloud1, pcloud2, dist_thresh=10):
    """
    Compute support matrix for two point clouds.
    The matrix indicates if the distance between two points is less than a threshold.

    Parameters
    ----------
    pcloud1 : torch.Tensor
        Point cloud 1. Size B x N x 3.
    pcloud2 : torch.Tensor
        Point cloud 2. Size B x M x 3.
    dist_thresh:
        Threshold on the Euclidean distance between points.

    Returns
    -------
    torch.Tensor
        Support matrix. Size B x N x M.

    """
    # Squared l2 distance between points points of both point clouds
    distance_matrix = torch.sum(pcloud1 ** 2, -1, keepdim=True)
    distance_matrix = distance_matrix + torch.sum(pcloud2 ** 2, -1, keepdim=True).transpose(1, 2)
    distance_matrix = distance_matrix - 2 * torch.bmm(pcloud1, pcloud2.transpose(1, 2))

    # Force transport to be zero for points further than 10 m apart
    support_mat = (distance_matrix < dist_thresh ** 2).float()

    return support_mat


def get_similarity_matrix(feature1, feature2):
    """
    Cosine similarity between point cloud features

    Parameters
    ----------
    feature1 : torch.Tensor
        Feature for points cloud 1. Used to computed transport cost. Size B x N x C.
    feature2 : torch.Tensor
        Feature for points cloud 2. Used to computed transport cost. Size B x M x C.

    Returns
    -------
    torch.Tensor
        Feature similarity matrix. Size B x N x M.
    """
    # Normalize features
    feature1_normalized = feature1 / (feature1.norm(dim=-1, p=2, keepdim=True) + 1e-8)
    feature2_normalized = feature2 / (feature2.norm(dim=-1, p=2, keepdim=True) + 1e-8)

    # Compute feature similarity
    sim_mat = torch.bmm(feature1_normalized, feature2_normalized.transpose(1, 2))

    return sim_mat


def normalize_mat(mat, mat_normalization, dim=None):
    """
    The method to normalize the input matrix to be like a statistical matrix.
    """
    if dim is None:
        dim = 1 if len(mat.shape) == 3 else 0

    if mat_normalization == "none":
        return mat
    if mat_normalization == "softmax":
        return F.softmax(mat, dim=dim)
    raise NameError


def get_s_t_topk(mat, k, s_only=False, nn_idx=None):
    """
    Get nearest neighbors per point (similarity value and index) for source and target shapes

    Args:
        mat (BxNsxNb Tensor): Similarity matrix
        k: Number of neighbors per point
        s_only: Whether to get neighbors only for the source point cloud or also for the target point cloud.
        nn_idx: An optional pre-computed nearest neighbor indices.
    """
    if nn_idx is not None:
        assert s_only, "Pre-computed nearest neighbor indices is allowed for the source point cloud only."
        s_nn_idx = nn_idx
        s_nn_val = mat.gather(dim=2, index=nn_idx)
        t_nn_val = t_nn_idx = None
    else:
        s_nn_val, s_nn_idx = mat.topk(k=min(k, mat.shape[2]), dim=2)

        if not s_only:
            t_nn_val, t_nn_idx = mat.topk(k=k, dim=1)

            t_nn_val = t_nn_val.transpose(2, 1)
            t_nn_idx = t_nn_idx.transpose(2, 1)
        else:
            t_nn_val = None
            t_nn_idx = None

    return s_nn_val, s_nn_idx, t_nn_val, t_nn_idx


def get_s_t_neighbors(k, mat, sim_normalization, s_only=False, ignore_first=False, nn_idx=None):
    s_nn_sim, s_nn_idx, t_nn_sim, t_nn_idx = get_s_t_topk(mat, k, s_only=s_only, nn_idx=nn_idx)
    if ignore_first:
        s_nn_sim, s_nn_idx = s_nn_sim[:, :, 1:], s_nn_idx[:, :, 1:]

    s_nn_weight = normalize_mat(s_nn_sim, sim_normalization, dim=2)

    if not s_only:
        if ignore_first:
            t_nn_sim, t_nn_idx = t_nn_sim[:, :, 1:], t_nn_idx[:, :, 1:]

        t_nn_weight = normalize_mat(t_nn_sim, sim_normalization, dim=2)
    else:
        t_nn_weight = None

    return s_nn_weight, s_nn_sim, s_nn_idx, t_nn_weight, t_nn_sim, t_nn_idx


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)

    return idx


def get_graph_feature(x, k, idx=None, only_intrinsic='neighs', permute_feature=True):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    else:
        if len(idx.shape) == 2:
            idx = idx.unsqueeze(0).repeat(batch_size, 1, 1)
        idx = idx[:, :, :k]
        k = min(k, idx.shape[-1])

    num_idx = idx.shape[1]

    idx_base = torch.arange(0, batch_size, device=idx.device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.contiguous()
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims) -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_idx, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    if only_intrinsic == 'true':
        feature = feature - x
    elif only_intrinsic == 'neighs':
        feature = feature
    elif only_intrinsic == 'concat':
        feature = torch.cat((feature, x), dim=3)
    else:
        feature = torch.cat((feature - x, x), dim=3)

    if permute_feature:
        feature = feature.permute(0, 3, 1, 2).contiguous()

    return feature


def reconstruct(pos, nn_idx, nn_weight, k):
    nn_pos = get_graph_feature(pos.transpose(1, 2), k=k, idx=nn_idx, only_intrinsic='neighs', permute_feature=False)
    nn_weighted = nn_pos * nn_weight.unsqueeze(dim=3)
    recon = torch.sum(nn_weighted, dim=2)

    return recon
