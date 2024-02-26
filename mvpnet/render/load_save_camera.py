import numpy as np
import open3d as o3d


def save_view_point(mesh, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=640, height=480)
    vis.add_geometry(mesh)
    ctr = vis.get_view_control()
    ctr.change_field_of_view(step=-10)
    # # zoom in z axis
    # param = ctr.convert_to_pinhole_camera_parameters()
    # param_array = np.array(param.extrinsic)
    # param_array[2][3] *= 0.9
    # param.extrinsic = param_array
    # ctr.convert_from_pinhole_camera_parameters(param)

    vis.run()  # user changes the view and press "q" to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(filename, param)
    print(vis.get_view_control().get_field_of_view())
    vis.destroy_window()


def load_view_point(mesh, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=640, height=480)
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters(filename)
    vis.add_geometry(mesh)
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    mesh = o3d.io.read_triangle_mesh("/mnt/disk/scannet3D/val/scene0011_00_vh_clean_2.ply")
    save_view_point(mesh, "viewpoint1.json")
    # load_view_point(mesh, "viewpoint1.json")