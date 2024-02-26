from glob import glob
from os.path import join

import numpy as np
import plyfile
import open3d as o3d

mesh_path = '/mnt/disk/scannet3D/val'
certainty_path = 'result/origin'
scenes = sorted(glob(join(mesh_path, '*_vh_clean_2.ply')))
NUM_X = 5
NUM_Y = 5
NUM_Z = 2
VIEW_NUM = 3
for scene in scenes:
    scene_id = scene.split('/')[-1][0:12]
    # mesh = plyfile.PlyData.read(scene)
    mesh = o3d.io.read_triangle_mesh(scene)
    coords = np.ascontiguousarray(mesh.vertices)

    certainty = np.loadtxt(join(certainty_path, scene_id + 'certainty.txt'))
    print(scene_id)
    print(coords[:, 0].max(), coords[:, 0].min())
    print(coords[:, 1].max(), coords[:, 1].min())
    print(coords[:, 2].max(), coords[:, 2].min())

    x_min = coords[:, 0].min()
    y_min = coords[:, 1].min()
    z_min = coords[:, 2].min()
    x_max = coords[:, 0].max() + 0.01
    y_max = coords[:, 1].max() + 0.01
    z_max = coords[:, 2].max() + 0.01
    x_length = (x_max - x_min) / NUM_X
    y_length = (y_max - y_min) / NUM_Y
    z_length = (z_max - z_min) / NUM_Z
    coords[:, 0] = ((coords[:, 0] - x_min) / x_length)
    coords[:, 1] = ((coords[:, 1] - y_min) / y_length)
    coords[:, 2] = ((coords[:, 2] - z_min) / z_length)
    coords = np.floor(coords).astype(int)
    glob_certainty = np.zeros((NUM_X, NUM_Y, NUM_Z))
    vertex_number = np.zeros((NUM_X, NUM_Y, NUM_Z), dtype=int)
    for i, coord in enumerate(coords):
        glob_certainty[coord[0], coord[1], coord[2]] += certainty[i]
        vertex_number[coord[0], coord[1], coord[2]] += 1
    vertex_number[vertex_number == 0] = 1
    glob_certainty = glob_certainty / vertex_number
    glob_certainty[glob_certainty == 0] = 100
    render_list = []
    certainty_dict = {}
    for index, value in np.ndenumerate(glob_certainty):
        certainty_dict[index] = value
        print(index, value)
        print(glob_certainty[index])
    certainty_dict = sorted(certainty_dict.items(), key=lambda d: d[1])

    vis = o3d.visualization.Visualizer()
    mesh = o3d.io.read_triangle_mesh(scene)
    vis.add_geometry(mesh)
    vis.create_window(visible=True, width=640, height=480)
    ctr = vis.get_view_control()
    param = ctr.convert_to_pinhole_camera_parameters()

    for coord in certainty_dict[:VIEW_NUM]:
        x = coord[0][0]
        y = coord[0][1]
        z = coord[0][2]
        x = (x  + 0.5) * x_length + x_min
        y = (y + 0.5) * y_length + y_min
        z = (z + 0.5) * z_length + z_min

        extrinsic = np.zeros((4, 4))

        param = ctr.convert_to_pinhole_camera_parameters()
        param.extrinsic = extrinsic
        param.intrinsic.set_intrinsics(640, 480, 240, 240, 319.5, 239.5)
        ctr.convert_from_pinhole_camera_parameters(param)
        # vis.poll_events()
        # vis.update_renderer()
        vis.run()
        vis.capture_screen_image('test1.jpg', True)
        vis.destroy_window()


def matrix_from_z(z_axis):
    x_axis0 = 1
    x_axis2 = 0  # parallel to ground
    x_axis1 = -z_axis[0] / z_axis[1]
    y_axis2 = 1
    y_axis1 = z_axis[2] / (x_axis1 * z_axis[0] - z_axis[1])
    y_axis0 = x_axis1 * z_axis[2] / (z_axis[1] - x_axis1 * z_axis[0])
    x_axis = np.array([x_axis0, x_axis1, x_axis2])
    y_axis = np.array([y_axis0, y_axis1, y_axis2])
    x_norm = np.linalg.norm(x_axis)
    y_norm = np.linalg.norm(y_axis)
    z_norm = np.linalg.norm(z_axis)
    x_axis = x_axis / x_norm
    y_axis = y_axis / y_norm
    z_axis = z_axis / z_norm
    rotation_matrix = np.array([x_axis, y_axis, z_axis])
    return rotation_matrix

