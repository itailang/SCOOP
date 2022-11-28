import os
import sys
import argparse

# add path
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_dir not in sys.path:
    sys.path.append(project_dir)

from tools.utils import create_dir, load_data
from tools.vis_utils import plot_pc_list, plot_pcs, save_pc_plot

COLOR_RED = (1, 0, 0)
COLOR_GREEN = (0, 1, 0)
COLOR_BLUE = (0, 0, 1)
COLOR_PURPLE = (127/255., 0, 1)


def vis_result(res_dir, res_srt, view_dict, save_dir, save_plot=True, show_plot=False, size=(1600, 800)):
    res_path = os.path.join(res_dir, "%s_res.npz" % res_srt)
    pc1, pc2, gt_mask, gt_flow, est_flow, corr_conf, pc1_warped_gt, pc1_warped_est = load_data(res_path)

    # pc1, pc2
    pc_list = [pc1, pc2]
    color_list = [COLOR_RED, COLOR_GREEN]
    plot_pc_list(pc_list, color_list, view_dict=view_dict, get_view=False, fig=None, size=size)

    save_path = os.path.join(save_dir, "%s_01_pc1_pc2.png" % res_srt)
    save_pc_plot(save_path, save=save_plot, show=show_plot)

    # pc1, pc1_warped_est
    pc_list = [pc1, pc1_warped_est]
    color_list = [COLOR_RED, COLOR_BLUE]
    plot_pc_list(pc_list, color_list, view_dict=view_dict, get_view=False, fig=None, size=size)

    save_path = os.path.join(save_dir, "%s_02_pc1_pc1_warped_est.png" % res_srt)
    save_pc_plot(save_path, save=save_plot, show=show_plot)

    # pc2, pc1_warped_est
    pc_list = [pc2, pc1_warped_est]
    color_list = [COLOR_GREEN, COLOR_BLUE]
    plot_pc_list(pc_list, color_list, view_dict=view_dict, get_view=False, fig=None, size=size)

    save_path = os.path.join(save_dir, "%s_03_pc2_pc1_warped_est.png" % res_srt)
    save_pc_plot(save_path, save=save_plot, show=show_plot)

    # pc1, pc2, pc1_warped_est
    pc_list = [pc1, pc2, pc1_warped_est]
    color_list = [COLOR_RED, COLOR_GREEN, COLOR_BLUE]
    plot_pc_list(pc_list, color_list, view_dict=view_dict, get_view=False, fig=None, size=size)

    save_path = os.path.join(save_dir, "%s_04_pc1_pc2_pc1_warped_est.png" % res_srt)
    save_pc_plot(save_path, save=save_plot, show=show_plot)

    # pc1, pc1_warped_gt
    pc_list = [pc1, pc1_warped_gt]
    color_list = [COLOR_RED, COLOR_PURPLE]
    plot_pc_list(pc_list, color_list, view_dict=view_dict, get_view=False, fig=None, size=size)

    save_path = os.path.join(save_dir, "%s_05_pc1_pc1_warped_gt.png" % res_srt)
    save_pc_plot(save_path, save=save_plot, show=show_plot)

    # pc2, pc1_warped_gt
    pc_list = [pc2, pc1_warped_gt]
    color_list = [COLOR_GREEN, COLOR_PURPLE]
    plot_pc_list(pc_list, color_list, view_dict=view_dict, get_view=False, fig=None, size=size)

    save_path = os.path.join(save_dir, "%s_06_pc2_pc1_warped_gt.png" % res_srt)
    save_pc_plot(save_path, save=save_plot, show=show_plot)

    # pc1, pc2, pc1_warped_gt
    pc_list = [pc1, pc2, pc1_warped_gt]
    color_list = [COLOR_RED, COLOR_GREEN, COLOR_PURPLE]
    plot_pc_list(pc_list, color_list, view_dict=view_dict, get_view=False, fig=None, size=size)

    save_path = os.path.join(save_dir, "%s_07_pc1_pc2_pc1_warped_gt.png" % res_srt)
    save_pc_plot(save_path, save=save_plot, show=show_plot)

    # pc1, est_flow (quiver), pc2
    color_dict = {"pc1_c": COLOR_RED, "pc2_c": COLOR_GREEN, "flow_c": COLOR_BLUE}
    plot_pcs(pc1, pc2, flow=est_flow, color_dict=color_dict, view_dict=view_dict, get_view=False, size=size)

    save_path = os.path.join(save_dir, "%s_08_pc1_flow_est_pc2.png" % res_srt)
    save_pc_plot(save_path, save=save_plot, show=show_plot)

    # pc1, gt_flow (quiver), pc2
    color_dict = {"pc1_c": COLOR_RED, "pc2_c": COLOR_GREEN, "flow_c": COLOR_PURPLE}
    plot_pcs(pc1, pc2, flow=gt_flow, color_dict=color_dict, view_dict=view_dict, get_view=False, size=size)

    save_path = os.path.join(save_dir, "%s_09_pc1_flow_gt_pc2.png" % res_srt)
    save_pc_plot(save_path, save=save_plot, show=show_plot)


if __name__ == "__main__":
    # Args
    parser = argparse.ArgumentParser(description="Visualize SCOOP.")
    parser.add_argument("--res_dir", type=str, default="./../pretrained_models/kitti_v_100_examples/pc_res", help="Point cloud results directory.")
    parser.add_argument("--res_idx", type=int, default=1, help="Index of the result to visualize.")
    args = parser.parse_args()

    view_dict = {'azimuth': 21.245700205624324, 'elevation': 99.80932242859983, 'distance': 13.089033739603677,
                 'focalpoint': [0.63038374, -1.79234603, 10.63751715], 'roll': 0.7910293664830116}

    save_dir = create_dir(os.path.join(args.res_dir, "vis"))
    vis_result(args.res_dir, "%06d" % args.res_idx, view_dict, save_dir, save_plot=True, show_plot=False, size=(1600, 800))
