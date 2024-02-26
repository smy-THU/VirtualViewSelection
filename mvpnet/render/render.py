import time

import open3d as o3d
import os, glob
import numpy as np


def render(ply_file, out_path, name):
    scene_id = ply_file[-27:-15]
    path = os.path.join(out_path, scene_id)
    if not os.path.isdir(path):
        os.makedirs(path)
    color_path = os.path.join(path, 'color')
    if not os.path.isdir(color_path):
        os.makedirs(color_path)
    depth_path = os.path.join(path, 'depth')
    if not os.path.isdir(depth_path):
        os.makedirs(depth_path)
    pose_path = os.path.join(path, 'pose')
    if not os.path.isdir(pose_path):
        os.makedirs(pose_path)
    out_image_file = os.path.join(color_path, name+'.jpg')
    out_depth_file = os.path.join(depth_path, name+'.png')
    out_pose_file = os.path.join(pose_path, name+'.json')

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=640, height=480)
    ctr = vis.get_view_control()
    mesh = o3d.io.read_triangle_mesh(ply_file)
    vis.add_geometry(mesh)
    # vis.get_render_option().load_from_json("./renderoption.json")
    # ctr.convert_from_pinhole_camera_parameters(camera_param)

    ctr.change_field_of_view(90)

    param = ctr.convert_to_pinhole_camera_parameters()
    param_array = np.array(param.extrinsic)
    param_array[2][3] *= 0.9
    param.extrinsic = param_array
    ctr.convert_from_pinhole_camera_parameters(param)

    ctr.rotate(-200, -200)
    ctr.translate(-20, -20)

    # vis.run()
    vis.poll_events()
    vis.update_renderer()

    # time.sleep(1)

    # buffer = vis.capture_screen_float_buffer(True)
    # image = np.asarray(buffer)
    # buffer = vis.capture_depth_float_buffer(False)
    # depth = np.asarray(buffer)
    vis.capture_depth_image(out_depth_file, True)

    o3d.io.write_pinhole_camera_parameters(out_pose_file, vis.get_view_control().convert_to_pinhole_camera_parameters())

    vis.destroy_window()
    

def render_global():
    ply_list = glob.glob("/mnt/disk/scannet3D/val/*2.ply")
    out_path = '/mnt/disk/scannet3D/render'
    #camera_param = o3d.io.read_pinhole_camera_parameters('./viewpoint0.json')
    name = '02'
    for ply_file in ply_list:
        s_time = time.time()
        render(ply_file, out_path, name)
        e_time = time.time()
        print(e_time-s_time)
    print("complete")


if __name__ == "__main__":
    render_global()
