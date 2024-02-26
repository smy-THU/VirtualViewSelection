import os
import open3d as o3d
import numpy as np


def render_from_viewpoints(ply_file, scene_path):
    viewpoints_path = os.path.join(scene_path, 'viewpoint')
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=640, height=480, visible=False)
    vis.get_render_option().load_from_json("render/renderoption.json")
    ctr = vis.get_view_control()
    mesh = o3d.io.read_triangle_mesh(ply_file)
    vis.add_geometry(mesh)
    for viewpoint_f in sorted(os.listdir(viewpoints_path), key=lambda x: int(x[:-5])):
        frame = viewpoint_f[:-5]
        param = o3d.io.read_pinhole_camera_parameters(os.path.join(viewpoints_path, viewpoint_f))
        ctr.convert_from_pinhole_camera_parameters(param)
        vis.poll_events()
        vis.update_renderer()
        color_path = os.path.join(scene_path, 'render_color')
        depth_path = os.path.join(scene_path, 'render_depth')
        if not os.path.isdir(color_path):
            os.makedirs(color_path)
        if not os.path.isdir(depth_path):
            os.makedirs(depth_path)
        vis.capture_depth_image(os.path.join(depth_path, frame + '.png'), True)
        vis.capture_screen_image(os.path.join(color_path, frame + '.jpg'), True)
    vis.destroy_window()
    print(ply_file, 'complete')


if __name__ == "__main__":
    scans_path = '/home/smy/scannet/scans'
    view_scans_path = '/mnt/disk/scannet3D/2D'
    for scene in sorted(os.listdir(scans_path)):
        ply_file = os.path.join(scans_path, scene, scene+'_vh_clean_2.ply')
        # camera_trajectory = os.path.join(tra_scans_path, scene, 'camera_trajectory.json')
        scene_path = os.path.join(view_scans_path, scene)
        render_from_viewpoints(ply_file, scene_path)
