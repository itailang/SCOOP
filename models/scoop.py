import torch
from tools import ot, reconstruction as R
from models.graph import Graph
from models.gconv import SetConv


class SCOOP(torch.nn.Module):
    def __init__(self, args):
        """
        Construct a model that, once trained, estimate the scene flow between
        two point clouds.

        Parameters
        ----------
        args.nb_iter : int
            Number of iterations to unroll in the Sinkhorn algorithm.

        """

        super(SCOOP, self).__init__()

        # Hand-chosen parameters. Define the number of channels.
        n = 32

        # OT parameters
        # Number of unrolled iterations in the Sinkhorn algorithm
        self.nb_iter = args.nb_iter
        # Mass regularisation
        self.gamma = torch.nn.Parameter(torch.zeros(1))
        # Entropic regularisation
        self.epsilon = torch.nn.Parameter(torch.zeros(1))

        # Architecture parameters
        self.nb_neigh_cross_recon = args.nb_neigh_cross_recon

        try:
            self.linear_corr_conf = args.linear_corr_conf
        except AttributeError:
            self.linear_corr_conf = 0

        # Feature extraction
        self.feat_conv1 = SetConv(3, n)
        self.feat_conv2 = SetConv(n, 2 * n)
        self.feat_conv3 = SetConv(2 * n, 4 * n)

    def get_features(self, pcloud, nb_neighbors=32):
        """
        Compute deep features for each point of the input point cloud. These
        features are used to compute the transport cost matrix between two
        point clouds.
        
        Parameters
        ----------
        pcloud : torch.Tensor
            Input point cloud of size B x N x 3
        nb_neighbors : int
            Number of nearest neighbors for each point.

        Returns
        -------
        x : torch.Tensor
            Deep features for each point. Size B x N x 128
        graph : scoop.models.graph.Graph
            Graph build on input point cloud containing list of nearest 
            neighbors (NN) and edge features (relative coordinates with NN).

        """

        graph = Graph.construct_graph(pcloud, nb_neighbors)
        x = self.feat_conv1(pcloud, graph)
        x = self.feat_conv2(x, graph)
        x = self.feat_conv3(x, graph)

        return x, graph

    def get_recon_flow(self, pclouds, feats):
        feats_0, feats_1 = feats[0], feats[1]

        # Reconstructed target point cloud
        transport_cross, similarity_cross = ot.sinkhorn(
            feats_0,
            feats_1,
            pclouds[0],
            pclouds[1],
            epsilon=torch.exp(self.epsilon) + 0.03,
            gamma=torch.exp(self.gamma),
            max_iter=self.nb_iter,
        )

        if self.nb_neigh_cross_recon > 0:
            source_cross_nn_weight, _, source_cross_nn_idx, _, _, _ = \
                R.get_s_t_neighbors(self.nb_neigh_cross_recon, transport_cross, sim_normalization="none", s_only=True)

            # Target point cloud cross reconstruction
            cross_weight_sum = source_cross_nn_weight.sum(-1, keepdim=True)
            source_cross_nn_weight_normalized = source_cross_nn_weight / (cross_weight_sum + 1e-8)
            target_cross_recon = R.reconstruct(pclouds[1], source_cross_nn_idx, source_cross_nn_weight_normalized, self.nb_neigh_cross_recon)

            # Matching probability
            cross_nn_sim, _, _, _ = R.get_s_t_topk(similarity_cross, self.nb_neigh_cross_recon, s_only=True, nn_idx=source_cross_nn_idx)
            nn_sim_weighted = cross_nn_sim * source_cross_nn_weight_normalized
            nn_sim_weighted = torch.sum(nn_sim_weighted, dim=2)
            if self.linear_corr_conf:
                corr_conf = (nn_sim_weighted + 1) / 2
            else:
                corr_conf = torch.clamp_min(nn_sim_weighted, 0.0)
        else:
            row_sum = transport_cross.sum(-1, keepdim=True)
            target_cross_recon = (transport_cross @ pclouds[1]) / (row_sum + 1e-8)
            corr_conf = None

        # Estimate flow from target cross reconstruction
        recon_flow = target_cross_recon - pclouds[0]

        return recon_flow, corr_conf, target_cross_recon

    def forward(self, pclouds):
        """
        Estimate scene flow between two input point clouds.

        Parameters
        ----------
        pclouds : (torch.Tensor, torch.Tensor)
            List of input point clouds (pc1, pc2). pc1 has size B x N x 3.
            pc2 has size B x M x 3.

        Returns
        -------
        est_flow : torch.Tensor
            Estimated scene flow of size B x N x 3.

        """

        # Extract features
        feats_0, graph = self.get_features(pclouds[0])
        feats_1, _ = self.get_features(pclouds[1])
        feats = [feats_0, feats_1]

        # Get reconstruction-based flow
        recon_flow, corr_conf, target_cross_recon = self.get_recon_flow(pclouds, feats)

        return recon_flow, corr_conf, target_cross_recon, graph
