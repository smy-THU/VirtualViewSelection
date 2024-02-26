import os
import json


def gen_trajectory(viewpoints_path, out_file):
    class_name = 'PinholeCameraTrajectory'
    parameters = []
    for viewpoint_f in sorted(os.listdir(viewpoints_path), key=lambda x: int(x[:-5])):
        with open(os.path.join(viewpoints_path, viewpoint_f)) as f:
            viewpoint = json.load(f)
        parameters.append(viewpoint)
    version_major = 1
    version_minor = 0
    camera_trajectory = {
        'class_name': class_name,
        'parameters': parameters,
        'version_major': version_major,
        'version_minor': version_minor
    }
    with open(out_file, 'w') as fo:
        json.dump(camera_trajectory, fo)


if __name__ == '__main__':
    scans_path = '/mnt/disk/scannet3D/2D/'
    for scene in os.listdir(scans_path):
        viewpoints_path = os.path.join(scans_path, scene, 'viewpoint')
        out_file = os.path.join(os.path.join(scans_path, scene, 'camera_trajectory.json'))
        gen_trajectory(viewpoints_path, out_file)
