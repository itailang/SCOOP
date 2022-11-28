import os
import numpy as np
from .generic import SceneFlowDataset


class Kitti(SceneFlowDataset):
    def __init__(self, root_dir, nb_points, all_points, mode):
        """
        Construct the KITTI scene flow datatset as in:
        Gu, X., Wang, Y., Wu, C., Lee, Y.J., Wang, P., HPLFlowNet: Hierarchical
        Permutohedral Lattice FlowNet for scene ﬂow estimation on large-scale 
        point clouds. IEEE Conf. Computer Vision and Pattern Recognition 
        (CVPR). pp. 3254–3263 (2019) 

        Parameters
        ----------
        root_dir : str
            Path to root directory containing the datasets.
        nb_points : int
            Maximum number of points in point clouds.
        all_points : bool
            Whether to use all point in the point cloud (in chucks of nb_points) or only nb_points.
        mode : str
            'train': training dataset.

            'val': validation dataset.

            'test': test dataset

            'all': all dataset

        """

        super(Kitti, self).__init__(nb_points, all_points)
        self.mode = mode
        self.root_dir = root_dir
        self.paths = self.make_dataset()
        self.filename_curr = ""

    def __len__(self):

        return len(self.paths)

    def make_dataset(self):
        """
        Find and filter out paths to all examples in the dataset. 
        
        """

        #
        root = os.path.realpath(os.path.expanduser(self.root_dir))
        all_paths = sorted(os.walk(root))
        useful_paths = [item[0] for item in all_paths if len(item[1]) == 0]
        assert len(useful_paths) == 200, "Problem with size of kitti dataset"

        # Mapping / Filtering of scans as in HPLFlowNet code
        mapping_path = os.path.join(os.path.dirname(__file__), "KITTI_mapping.txt")
        with open(mapping_path) as fd:
            lines = fd.readlines()
            lines = [line.strip() for line in lines]
        useful_paths = [
            path for path in useful_paths if lines[int(os.path.split(path)[-1])] != ""
        ]

        useful_paths = np.array(useful_paths)
        len_dataset = len(useful_paths)

        # the train indices was randomly selected by:
        # train_idx = self.make_subset_idx(total_examples=len_dataset, nb_examples=92, seed=42)
        train_idx = [0, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26, 27, 28, 30, 31, 33,
                     34, 35, 36, 39, 40, 42, 43, 44, 45, 47, 49, 51, 53, 55, 56, 60, 62, 64, 65, 66, 67, 68, 69, 70,
                     73, 76, 77, 78, 80, 81, 82, 83, 84, 85, 86, 89, 93, 94, 95, 96, 97, 98, 101, 105, 108, 109, 110,
                     111, 112, 113, 114, 115, 117, 118, 122, 123, 124, 126, 128, 130, 131, 133, 135, 137, 138, 140]

        test_idx = [i for i in range(len_dataset) if i not in train_idx]
        val_idx = test_idx

        if self.mode == "train":
            useful_paths_train = useful_paths[train_idx]
            train_size = 92
            assert len(useful_paths_train) == train_size, "Problem with size of kitti train dataset"
            useful_paths = useful_paths_train

        elif self.mode == "val":
            useful_paths_val = useful_paths[val_idx]
            val_size = 50
            assert len(useful_paths_val) == val_size, "Problem with size of kitti validation dataset"
            useful_paths = useful_paths_val

        elif self.mode == "test":
            useful_paths_test = useful_paths[test_idx]
            assert len(useful_paths_test) == 50, "Problem with size of kitti test dataset"
            useful_paths = useful_paths_test

        elif self.mode == "all":
            assert len(useful_paths) == 142, "Problem with size of kitti dataset"
        else:
            raise ValueError("Mode " + str(self.mode) + "unknown.")

        useful_paths = list(useful_paths)

        return useful_paths

    def make_subset_idx(self, total_examples, nb_examples, seed=42):
        np.random.seed(seed)
        idx_perm = np.random.permutation(total_examples)
        idx_sel = np.sort(idx_perm[:nb_examples])

        return idx_sel

    def load_sequence(self, idx):
        """
        Load a sequence of point clouds.

        Parameters
        ----------
        idx : int
            Index of the sequence to load.

        Returns
        -------
        sequence : list(np.array, np.array)
            List [pc1, pc2] of point clouds between which to estimate scene 
            flow. pc1 has size n x 3 and pc2 has size m x 3.
            
        ground_truth : list(np.array, np.array)
            List [mask, flow]. mask has size n x 1 and pc1 has size n x 3. 
            flow is the ground truth scene flow between pc1 and pc2. mask is 
            binary with zeros indicating where the flow is not valid/occluded.

        """

        # Load data
        self.filename_curr = self.paths[idx]
        sequence = [np.load(os.path.join(self.paths[idx], "pc1.npy"))]
        sequence.append(np.load(os.path.join(self.paths[idx], "pc2.npy")))

        # Remove ground points
        is_ground = np.logical_and(sequence[0][:, 1] < -1.4, sequence[1][:, 1] < -1.4)
        not_ground = np.logical_not(is_ground)
        sequence = [sequence[i][not_ground] for i in range(2)]

        # Remove points further than 35 meter away as in HPLFlowNet code
        is_close = np.logical_and(sequence[0][:, 2] < 35, sequence[1][:, 2] < 35)
        sequence = [sequence[i][is_close] for i in range(2)]

        # Scene flow
        ground_truth = [
            np.ones_like(sequence[0][:, 0:1]),
            sequence[1] - sequence[0],
        ]  # [Occlusion mask, scene flow]

        return sequence, ground_truth
