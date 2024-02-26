import os
import json
import numpy as np


def gen_viewpoint(pose_file, intrinsic_file, out_file, height=480, width=640):
    class_name = 'PinholeCameraParameters'
    pose = np.asarray(
        [[float(x[0]), float(x[1]), float(x[2]), float(x[3])] for x in
         (x.split(" ") for x in open(pose_file).read().splitlines())]
    )
    extrin = np.linalg.inv(pose)
    judge_m =  np.isinf(pose)
    for inds, b in np.ndenumerate(judge_m):
        if b:
            print('skip', pose_file)
            return
    extrinsic = extrin.T.reshape(16).tolist()
    intrin = np.asarray(
        [[float(x[0]), float(x[1]), float(x[2]), float(x[3])] for x in
         (x.split(" ") for x in open(intrinsic_file).read().splitlines())]
    )
    intrin = intrin[:3, :3]
    intrin[0][0] *= width/1296
    intrin[1][1] *= height/968
    intrin[0][2] = width/2 - 0.5
    intrin[1][2] = height/2 - 0.5
    intrinsic_matrix = intrin.T.reshape(9).tolist()
    intrinsic = {'height': height,
                 'intrinsic_matrix': intrinsic_matrix,
                 'width': width}
    version_major = 1
    version_minor = 0
    viewpoint = {'class_name': class_name,
                 'extrinsic': extrinsic,
                 'intrinsic': intrinsic,
                 'version_major': version_major,
                 'version_minor': version_minor}
    with open(out_file, 'w') as f:
        json.dump(viewpoint, f)


def main():
    scannet_path = '/mnt/disk/scannet3D/2D'
    for scene in sorted(os.listdir(scannet_path)):
        intrinsic_file = os.path.join(scannet_path, scene, 'intrinsic', 'intrinsic_color.txt')
        pose_path = os.path.join(scannet_path, scene, 'pose')
        out_path = os.path.join(scannet_path, scene, 'viewpoint')
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        for file in os.listdir(pose_path):
            pose_file = os.path.join(pose_path, file)
            frame = file[:-4]
            out_file = os.path.join(out_path, frame + '.json')
            gen_viewpoint(pose_file, intrinsic_file, out_file)
        print(scene, 'complete')


if __name__ == '__main__':
    main()
