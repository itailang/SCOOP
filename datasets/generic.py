import torch
import numpy as np
from torch.utils.data import Dataset


class Batch:
    def __init__(self, batch):
        """
        Concatenate list of dataset.generic.SceneFlowDataset's item in batch 
        dimension.

        Parameters
        ----------
        batch : list
            list of dataset.generic.SceneFlowDataset's item.

        """

        self.data = {}
        batch_size = len(batch)
        for key in ["sequence", "ground_truth", "orig_size"]:
            self.data[key] = []
            for ind_seq in range(len(batch[0][key])):
                tmp = []
                for ind_batch in range(batch_size):
                    item = batch[ind_batch][key][ind_seq]
                    if len(item.shape) > 3:
                        tmp.append(item.reshape([-1, item.shape[-2], item.shape[-1]]))
                    else:
                        tmp.append(item)
                self.data[key].append(torch.cat(tmp, 0))

    def __getitem__(self, item):
        """
        Get 'sequence' or 'ground_truth' from the batch.
        
        Parameters
        ----------
        item : str
            Accept two keys 'sequence' or 'ground_truth'.

        Returns
        -------
        list(torch.Tensor, torch.Tensor)
            item='sequence': returns a list [pc1, pc2] of point clouds between 
            which to estimate scene flow. pc1 has size B x n x 3 and pc2 has 
            size B x m x 3.
            
            item='ground_truth': returns a list [mask, flow]. mask has size 
            B x n x 1 and flow has size B x n x 3. flow is the ground truth 
            scene flow between pc1 and pc2. flow is the ground truth scene 
            flow. mask is binary with zeros indicating where the flow is not 
            valid or occluded.
            
        """
        return self.data[item]

    def to(self, *args, **kwargs):

        for key in self.data.keys():
            self.data[key] = [d.to(*args, **kwargs) for d in self.data[key]]

        return self

    def pin_memory(self):

        for key in self.data.keys():
            self.data[key] = [d.pin_memory() for d in self.data[key]]

        return self


class SceneFlowDataset(Dataset):
    def __init__(self, nb_points, all_points=False):
        """
        Abstract constructor for scene flow datasets.
        
        Each item of the dataset is returned in a dictionary with two keys:
            (key = 'sequence', value=list(torch.Tensor, torch.Tensor)): 
            list [pc1, pc2] of point clouds between which to estimate scene 
            flow. pc1 has size 1 x n x 3 and pc2 has size 1 x m x 3.
            
            (key = 'ground_truth', value = list(torch.Tensor, torch.Tensor)): 
            list [mask, flow]. mask has size 1 x n x 1 and flow has size
            1 x n x 3. flow is the ground truth scene flow between pc1 and pc2.
            mask is binary with zeros indicating where the flow is not 
            valid/occluded.

        Parameters
        ----------
        nb_points : int
            Maximum number of points in point clouds: self.nb_points <= m, n.
        all_points : bool
            Whether to use all point in the point cloud (in chucks of nb_points) or only nb_points.

        """

        super(SceneFlowDataset, self).__init__()
        self.nb_points = nb_points
        self.all_points = all_points

    def __getitem__(self, idx):
        sequence, ground_truth, orig_size = self.to_torch(*self.subsample_points_rnd(*self.load_sequence(idx)))
        data = {"sequence": sequence, "ground_truth": ground_truth, "orig_size": orig_size}

        return data

    def to_torch(self, sequence, ground_truth, orig_size):
        """
        Convert numpy array and torch.Tensor.

        Parameters
        ----------
        sequence : list(np.array, np.array)
            List [pc1, pc2] of point clouds between which to estimate scene 
            flow. pc1 has size n x 3 and pc2 has size m x 3.
            
        ground_truth : list(np.array, np.array)
            List [mask, flow]. mask has size n x 1 and pc1 has size n x 3. 
            flow is the ground truth scene flow between pc1 and pc2. mask is 
            binary with zeros indicating where the flow is not valid/occluded.

        orig_size : list(np.array, np.array)
            List [n1, n2]. Original size of the point clouds.

        Returns
        -------
        sequence : list(torch.Tensor, torch.Tensor)
            List [pc1, pc2] of point clouds between which to estimate scene 
            flow. pc1 has size 1 x n x 3 and pc2 has size 1 x m x 3.
            
        ground_truth : list(torch.Tensor, torch.Tensor)
            List [mask, flow]. mask has size 1 x n x 1 and pc1 has size 
            1 x n x 3. flow is the ground truth scene flow between pc1 and pc2.
            mask is binary with zeros indicating where the flow is not 
            valid/occluded.

        """

        sequence = [torch.unsqueeze(torch.from_numpy(s), 0).float() for s in sequence]
        ground_truth = [torch.unsqueeze(torch.from_numpy(gt), 0).float() for gt in ground_truth]
        orig_size = [torch.unsqueeze(torch.from_numpy(os), 0) for os in orig_size]

        return sequence, ground_truth, orig_size

    def subsample_points_rnd(self, sequence, ground_truth):
        """
        Subsample point clouds randomly.

        Parameters
        ----------
        sequence : list(np.array, np.array)
            List [pc1, pc2] of point clouds between which to estimate scene 
            flow. pc1 has size 1 x N x 3 and pc2 has size 1 x M x 3.
            
        ground_truth : list(np.array, np.array)
            List [mask, flow]. mask has size 1 x N x 1 and flow has size
            1 x N x 3. flow is the ground truth scene flow between pc1 and pc2.
            mask is binary with zeros indicating where the flow is not 
            valid/occluded.

        Returns
        -------
        sequence : list(np.array, np.array)
            List [pc1, pc2] of point clouds between which to estimate scene 
            flow. pc1 has size 1 x n x 3 and pc2 has size 1 x m x 3. The n 
            points are chosen randomly among the N available ones. The m points
            are chosen randomly among the M available ones. If N, M >= 
            self.nb_point then n, m = self.nb_points. If N, M < 
            self.nb_point then n, m = N, M. 
            
        ground_truth : list(np.array, np.array)
            List [mask, flow]. mask has size 1 x n x 1 and flow has size
            1 x n x 3. flow is the ground truth scene flow between pc1 and pc2.
            mask is binary with zeros indicating where the flow is not 
            valid/occluded.

        """

        # Permute indices
        n1 = sequence[0].shape[0]
        idx1_perm = np.random.permutation(n1)

        n2 = sequence[1].shape[0]
        idx2_perm = np.random.permutation(n2)

        # Prepare indices for sampling
        if self.all_points:
            if self.nb_points == 1:
                idx1 = np.arange(n1)
                idx2 = np.arange(n2)
            else:
                n1_div_ceil = n1 // self.nb_points + int((n1 % self.nb_points) > 0)
                n1_ceil = n1_div_ceil * self.nb_points

                n2_div_ceil = n2 // self.nb_points + int((n2 % self.nb_points) > 0)
                n2_ceil = n2_div_ceil * self.nb_points

                # Take larger point cloud size, in order to have all point from both point clouds for evaluation
                n_ceil = n1_ceil > n2_ceil and n1_ceil or n2_ceil

                idx1 = np.concatenate([idx1_perm, idx1_perm[:(n_ceil - n1)]])
                idx2 = np.concatenate([idx2_perm, idx2_perm[:(n_ceil - n2)]])
        else:
            idx1 = idx1_perm[:self.nb_points]
            idx2 = idx2_perm[:self.nb_points]

        # Sample points in the first scan
        sequence[0] = sequence[0][idx1]
        ground_truth = [g[idx1] for g in ground_truth]

        # Sample point in the second scan
        sequence[1] = sequence[1][idx2]

        # Reshape data
        if self.all_points:
            sequence[0] = sequence[0].reshape([-1, self.nb_points, 3])
            ground_truth = [g.reshape([-1, self.nb_points, g.shape[1]]) for g in ground_truth]

            sequence[1] = sequence[1].reshape([-1, self.nb_points, 3])

        if self.nb_points == 1:
            sequence = [s.transpose(1, 0, 2) for s in sequence]
            ground_truth = [g.transpose(1, 0, 2) for g in ground_truth]

        orig_size = [np.array([n1], dtype=np.int32), np.array([n2], dtype=np.int32)]

        return sequence, ground_truth, orig_size

    def load_sequence(self, idx):
        """
        Abstract function to be implemented to load a sequence of point clouds.

        Parameters
        ----------
        idx : int
            Index of the sequence to load.

        Must return:
        -------
        sequence : list(np.array, np.array)
            List [pc1, pc2] of point clouds between which to estimate scene 
            flow. pc1 has size N x 3 and pc2 has size M x 3.
            
        ground_truth : list(np.array, np.array)
            List [mask, flow]. mask has size N x 1 and flow has size N x 3.
            flow is the ground truth scene flow between pc1 and pc2. mask is 
            binary with zeros indicating where the flow is not valid/occluded.

        """

        raise NotImplementedError
