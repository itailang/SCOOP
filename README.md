# SCOOP: Self-Supervised Correspondence and Optimization-Based Scene Flow
[[Project Page]](https://itailang.github.io/SCOOP/) [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/html/Lang_SCOOP_Self-Supervised_Correspondence_and_Optimization-Based_Scene_Flow_CVPR_2023_paper.html) [[Video]](https://www.youtube.com/watch?v=b8MVWGU7V4E) [[Slides]](./doc/slides.pdf) [[Poster]](./doc/poster.png)

Created by [Itai Lang](https://scholar.google.com/citations?user=q0bBhtsAAAAJ&hl=en/)<sup>1,2</sup>, [Dror Aiger](https://research.google/people/DrorAiger/)<sup>2</sup>, [Forrester Cole](http://people.csail.mit.edu/fcole/)<sup>2</sup>, [Shai Avidan](http://www.eng.tau.ac.il/~avidan/)<sup>1</sup>, and [Michael Rubinstein](http://people.csail.mit.edu/mrub/)<sup>2</sup>. <br>
<sup>1</sup>Tel Aviv University&nbsp;&nbsp;&nbsp;<sup>2</sup>Google Research

![scoop_result](./doc/scoop_result.gif)

## Abstract
Scene flow estimation is a long-standing problem in computer vision, where the goal is to find the 3D motion of a scene from its consecutive observations.
Recently, there have been efforts to compute the scene flow from 3D point clouds.
A common approach is to train a regression model that consumes source and target point clouds and outputs the per-point translation vector.
An alternative is to learn point matches between the point clouds concurrently with regressing a refinement of the initial correspondence flow.
In both cases, the learning task is very challenging since the flow regression is done in the free 3D space, and a typical solution is to resort to a large annotated synthetic dataset.

We introduce SCOOP, a new method for scene flow estimation that can be learned on a small amount of data without employing ground-truth flow supervision.
In contrast to previous work, we train a pure correspondence model focused on learning point feature representation and initialize the flow as the difference between a source point and its softly corresponding target point.
Then, in the run-time phase, we directly optimize a flow refinement component with a self-supervised objective, which leads to a coherent and accurate flow field between the point clouds.
Experiments on widespread datasets demonstrate the performance gains achieved by our method compared to existing leading techniques while using a fraction of the training data.

## Citation
If you find our work useful in your research, please consider citing:

	@InProceedings{lang2023scoop,
	  author = {Lang, Itai and Aiger, Dror and Cole, Forrester and Avidan, Shai and Rubinstein, Michael},
	  title = {{SCOOP: Self-Supervised Correspondence and Optimization-Based Scene Flow}},
	  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
	  pages = {5281--5290},
	  year = {2023}
	}

## Installation
The code has been tested with Python 3.6.13, PyTorch 1.6.0, CUDA 10.1, and cuDNN 7.6.5 on Ubuntu 16.04.

Clone this repository:
```bash
git clone https://github.com/itailang/SCOOP.git
cd SCOOP/
```

Create a conda environment: 
```bash
# create and activate a conda environment
conda create -n scoop python=3.6.13 --yes
conda activate scoop
```

Install required packages:
```bash
sh install_environment.sh
```

Compile the Chamfer Distance op, implemented by [Groueix _et al._](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch) The op is located under `auxiliary/ChamferDistancePytorch/chamfer3D` folder. The following compilation script uses a CUDA 10.1 path. If needed, modify script to point to your CUDA path. Then, use:
 ```bash
sh compile_chamfer_distance_op.sh
```

The compilation results should be created under `auxiliary/ChamferDistancePytorch/chamfer3D/build` folder.

## Usage

### Data
Create folders for the data:
```bash
mkdir ./data/
mkdir ./data/FlowNet3D/
mkdir ./data/HPLFlowNet/
```

We use the point cloud data version prepared Liu _et al._ from their work [FlowNet3D](https://openaccess.thecvf.com/content_CVPR_2019/html/Liu_FlowNet3D_Learning_Scene_Flow_in_3D_Point_Clouds_CVPR_2019_paper.html). Please follow their [code](https://github.com/xingyul/flownet3d) to acquire the data.
* Put the preprocessed FlyingThings3D dataset at `./data/FlowNet3D/data_processed_maxcut_35_20k_2k_8192/`. This data is denoted as FT3D<sub>o</sub>.
* Put the preprocessed KITTI dataset at `./data/flownet3d/kitti_rm_ground/`. This dataset is denoted as KITTI<sub>o</sub> and its subsets are denoted as KITTI<sub>v</sub> and KITTI<sub>t</sub>.  

We also use the point cloud data version prepared Gu _et al._ from their work [HPLFlowNet](https://openaccess.thecvf.com/content_CVPR_2019/html/Gu_HPLFlowNet_Hierarchical_Permutohedral_Lattice_FlowNet_for_Scene_Flow_Estimation_on_CVPR_2019_paper.html). Please follow their [code](https://github.com/laoreja/HPLFlowNet) to acquire the data.
* Put the preprocessed FlyingThings3D dataset at `./data/HPLFlowNet/FlyingThings3D_subset_processed_35m/`. This dataset is denoted as FT3D<sub>s</sub>.
* Put the preprocessed KITTI dataset at `./data/flownet3d/KITTI_processed_occ_final/`. This dataset is denoted as KITTI<sub>s</sub>.

Note that you may put the data elsewhere and create a symbolic link to the actual location. For example:
```bash
ln -s /path/to/the/actual/dataset/location ./data/FlowNet3D/data_processed_maxcut_35_20k_2k_8192  
```

### Training and Evaluation
Switch to the `scripts` folder:
```bash
cd ./scripts
```

#### FT3D<sub>o</sub> / KITTI<sub>o</sub>
To train a model on 1,800 examples from the train set of FT3D<sub>o</sub>, run the following command:
```bash
sh train_on_ft3d_o.sh
```

Evaluate this model on KITTI<sub>o</sub> with 2,048 point per point cloud using the following command:
```bash
sh evaluate_on_kitti_o.sh
```

The results will be saved to the file `./experiments/ft3d_o_1800_examples/log_evaluation_kitti_o.txt`.

Evaluate this model on KITTI<sub>o</sub> with all the points in the point clouds using the following command:
```bash
sh evaluate_on_kitti_o_all_point.sh
```

The results will be saved to the file `./experiments/ft3d_o_1800_examples/log_evaluation_kitti_o_all_points.txt`.

#### KITTI<sub>v</sub> / KITTI<sub>t</sub>
To train a model on the 100 examples of KITTI<sub>v</sub>, run the following command:
```bash
sh train_on_kitti_v.sh
```

Evaluate this model on KITTI<sub>t</sub> with 2,048 point per point cloud using the following command:
```bash
sh evaluate_on_kitti_t.sh
```

The results will be saved to the file `./experiments/kitti_v_100_examples/log_evaluation_kitti_t.txt`.

Evaluate this model on KITTI<sub>t</sub> with all the points in the point clouds using the following command:
```bash
sh evaluate_on_kitti_t_all_points.sh
```

The results will be saved to the file `./experiments/kitti_v_100_examples/log_evaluation_kitti_t_all_points.txt`.

#### FT3D<sub>s</sub> / KITTI<sub>s</sub>, FT3D<sub>s</sub> / FT3D<sub>s</sub>  
To train a model on 1,800 examples from the train set FT3D<sub>s</sub>, run the following command:
```bash
sh train_on_ft3d_s.sh
```

Evaluate this model on KITTI<sub>s</sub> with 8,192 point per point cloud using the following command:
```bash
sh evaluate_on_kitti_s.sh
```

The results will be saved to the file `./experiments/ft3d_s_1800_examples/log_evaluation_kitti_s.txt`.

Evaluate this model on the test set of FT3D<sub>s</sub> with 8,192 point per point cloud using the following command:
```bash
sh evaluate_on_ft3d_s.sh
```

The results will be saved to the file `./experiments/ft3d_s_1800_examples/log_evaluation_ft3d_s.txt`.

#### Visualization
First, save results for visualization by adding the flag `--save_pc_res 1` when running the evaluation script. For example, the [script for evaluating on KITTI<sub>t</sub>](./scripts/evaluate_on_kitti_t.sh).
The results will be saved to the folder `./experiments/kitti_v_100_examples/pc_res/`.

Then, select the scene index that you would like to visualize and run the visualization script. For example, visualizing scene index #1 from KITTI<sub>t</sub>:
```python
python visualize_scoop.py --res_dir ./../experiments/kitti_v_100_examples/pc_res --res_idx 1 
```

The visualizations will be saved to the folder `./experiments/kitti_v_100_examples/pc_res/vis/`.

### Evaluation with Pretrained Models

First, download our pretrained models with the following command:
```bash
bash download_pretrained_models.sh
```

The models (about 2MB) will be saved under `pretrained_models` folder in the following structure:
```
pretrained_models/
├── ft3d_o_1800_examples/model_e100.tar
├── ft3d_s_1800_examples/model_e060.tar
├── kitti_v_100_examples/model_e400.tar
```

Then, use the evaluation commands mentioned in section [Training and Evaluation](#training-and-Evaluation),
after changing the `experiments` folder in the evaluation scripts to the `pretrained_models` folder.

## License
This project is licensed under the terms of the MIT license (see the [LICENSE](./LICENSE) file for more details).

## Acknowledgment
Our code builds upon the code provided by [Puy _et al._](https://github.com/valeoai/FLOT), [Groueix _et al._](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch), [Liu _et al._](https://github.com/xingyul/flownet3d), and [Gu _et al._](https://github.com/laoreja/HPLFlowNet) We thank the authors for sharing their code.
