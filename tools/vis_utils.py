import os
import sys
from mayavi import mlab
from PIL import Image


def get_view_params(mlab_fig):
    azimuth, elevation, distance, focalpoint = mlab_fig.view()
    roll = mlab_fig.roll()

    view_dict = {"azimuth": azimuth, "elevation": elevation, "distance": distance, "focalpoint": focalpoint, "roll": roll}
    return view_dict


def plot_pc_list(pc_list, color_list, view_dict=None, get_view=False, fig=None, size=(1600, 800)):
    if fig is None:
        fig = mlab.figure(figure=None, bgcolor=(1, 1, 1), fgcolor=None, engine=None, size=size)

    for pc, color in zip(pc_list, color_list):
        if pc is not None:
            mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], scale_factor=0.05, color=color, figure=fig)

    if view_dict is not None:
        mlab.view(**view_dict)

    if get_view:
        view_dict = get_view_params(mlab)
        print(view_dict)

    return fig


def plot_pcs(pc1, pc2, mask=None, flow=None, color_dict=None, view_dict=None, get_view=False, fig=None, size=(1600, 800)):
    if color_dict is None:
        color_dict = {"pc1_c": (1, 0, 0), "pc2_c": (0, 1, 0), "flow_c": (0, 0, 1)}

    if fig is None:
        fig = mlab.figure(figure=None, bgcolor=(1, 1, 1), fgcolor=None, engine=None, size=size)

    num_points = len(pc1)
    if len(color_dict["pc1_c"]) == num_points:  # color per-point:
        p3d = mlab.points3d(pc1[:, 0], pc1[:, 1], pc1[:, 2], scale_factor=0.05, figure=fig)
        p3d.glyph.scale_mode = 'scale_by_vector'
        p3d.mlab_source.dataset.point_data.scalars = color_dict["pc1_c"]
    else:
        mlab.points3d(pc1[:, 0], pc1[:, 1], pc1[:, 2], scale_factor=0.05, color=color_dict["pc1_c"], figure=fig)

    if flow is not None:
        if mask is not None:
            pc1_plot = pc1[mask]
            flow_plot = flow[mask]
        else:
            pc1_plot = pc1
            flow_plot = flow
        mlab.quiver3d(pc1_plot[:, 0], pc1_plot[:, 1], pc1_plot[:, 2], flow_plot[:, 0], flow_plot[:, 1], flow_plot[:, 2], scale_factor=1, color=color_dict["flow_c"], line_width=2.0, figure=fig)
    if pc2 is not None:
        mlab.points3d(pc2[:, 0], pc2[:, 1], pc2[:, 2], scale_factor=0.05, color=color_dict["pc2_c"], figure=fig)

    if view_dict is not None:
        mlab.view(**view_dict)

    if get_view:
        view_dict = get_view_params(mlab)
        print(view_dict)

    return fig


def save_pc_plot(save_path, save=True, show=False):
    try:
        screenshot = mlab.screenshot()
    except:
        screenshot = mlab.screenshot()
    img = Image.fromarray(screenshot)

    if save:
        img.save(save_path)

    if show:
        mlab.show()
    else:
        mlab.close()
