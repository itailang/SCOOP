import os
import glob
import numpy as np
from .generic import SceneFlowDataset


class Kitti(SceneFlowDataset):
    def __init__(self, root_dir, nb_points, all_points, same_v_t_split, mode):
        """
        Construct the KITTI scene flow datatset as in:
        Liu, X., Qi, C.R., Guibas, L.J.: FlowNet3D: Learning scene ﬂow in 3D 
        point clouds. IEEE Conf. Computer Vision and Pattern Recognition 
        (CVPR). pp. 529–537 (2019) 

        Parameters
        ----------
        root_dir : str
            Path to root directory containing the datasets.
        nb_points : int
            Maximum number of points in point clouds.
        all_points : bool
            Whether to use all point in the point cloud (in chucks of nb_points) or only nb_points.
        same_v_t_split: bool
            Whether to use the same validation and test split.
        mode : str
            'train': training dataset.

            'val': validation dataset.

            'test': test dataset

            'all': all dataset

        """

        super(Kitti, self).__init__(nb_points, all_points)
        self.mode = mode
        self.root_dir = root_dir
        self.same_v_t_split = same_v_t_split
        self.filenames = self.make_dataset()
        self.filename_curr = ""

    def __len__(self):

        return len(self.filenames)

    def make_dataset(self):
        """
        Find and filter out paths to all examples in the dataset.

        """
        len_dataset = 150
        filenames_all = glob.glob(os.path.join(self.root_dir, "*.npz"))

        test_list = [1, 5, 7, 8, 10, 12, 15, 17, 20, 21, 24, 25, 29, 30, 31, 32, 34, 35, 36, 39, 40, 44, 45, 47, 48,
                     49, 50, 51, 53, 55, 56, 58, 59, 60, 70, 71, 72, 74, 76, 77, 78, 79, 81, 82, 88, 91, 93, 94, 95, 98]
        val_list = [4, 54, 73, 101, 102, 104, 115, 130, 136, 147]

        if self.same_v_t_split:
            val_list = test_list

        train_list = [i for i in range(len_dataset) if i not in test_list and i not in val_list]

        if self.mode == "train":
            filenames_train = [fn for fn in filenames_all if int(os.path.split(fn)[1].split(".")[0]) in train_list]
            train_size = 100 if self.same_v_t_split else 90
            assert len(filenames_train) == train_size, "Problem with size of kitti train dataset"
            filenames = filenames_train

        elif self.mode == "val":
            filenames_val = [fn for fn in filenames_all if int(os.path.split(fn)[1].split(".")[0]) in val_list]
            val_size = 50 if self.same_v_t_split else 10
            assert len(filenames_val) == val_size, "Problem with size of kitti validation dataset"
            filenames = filenames_val

        elif self.mode == "test":
            filenames_test = [fn for fn in filenames_all if int(os.path.split(fn)[1].split(".")[0]) in test_list]
            assert len(filenames_test) == 50, "Problem with size of kitti test dataset"
            filenames = filenames_test

        elif self.mode == "all":
            assert len(filenames_all) == 150, "Problem with size of kitti dataset"
            filenames = filenames_all
        else:
            raise ValueError("Mode " + str(self.mode) + "unknown.")

        return filenames

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
        self.filename_curr = self.filenames[idx]
        with np.load(self.filename_curr) as data:
            sequence = [data["pos1"][:, (1, 2, 0)], data["pos2"][:, (1, 2, 0)]]
            ground_truth = [
                np.ones_like(data["pos1"][:, 0:1]),
                data["gt"][:, (1, 2, 0)],
            ]

        # Restrict to 35m
        loc = sequence[0][:, 2] < 35
        sequence[0] = sequence[0][loc]
        ground_truth[0] = ground_truth[0][loc]
        ground_truth[1] = ground_truth[1][loc]
        loc = sequence[1][:, 2] < 35
        sequence[1] = sequence[1][loc]

        return sequence, ground_truth
