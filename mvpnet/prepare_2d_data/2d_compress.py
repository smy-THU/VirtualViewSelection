import os
import shutil
import sys
import cv2
import imageio
import numpy as np
from skimage import transform
import util
from tqdm import tqdm

scannet_path = "/home/smy/scannet/scans/"
output_path = '/mnt/disk/scannet3D/2D/'
height = 480
width = 640


def map_label_image(image, label_mapping):
    mapped = np.copy(image)
    for k, v in label_mapping.items():
        mapped[image == k] = v
    return mapped.astype(np.uint8)


def main():
    scenes = [d for d in sorted(os.listdir(scannet_path))]
    for scene in tqdm(scenes):
        if scene < 'scene0031_02':
            continue
        output_color_path = os.path.join(output_path, scene, 'color')
        if not os.path.isdir(output_color_path):
            os.makedirs(output_color_path)
        output_depth_path = os.path.join(output_path, scene, 'depth')
        if not os.path.isdir(output_depth_path):
            os.makedirs(output_depth_path)
        output_pose_path = os.path.join(output_path, scene, 'pose')
        if not os.path.isdir(output_pose_path):
            os.makedirs(output_pose_path)
        output_label_path = os.path.join(output_path, scene, 'label-filt')
        if not os.path.isdir(output_label_path):
            os.makedirs(output_label_path)

        intrinsic_file = os.path.join(scannet_path, scene, 'intrinsic', 'intrinsic_color.txt')
        output_intrinsic_path = os.path.join(output_path, scene, 'intrinsic')
        if not os.path.isdir(output_intrinsic_path):
            os.makedirs(output_intrinsic_path)
        shutil.copy(intrinsic_file, os.path.join(output_intrinsic_path, 'intrinsic_color.txt'))

        frames = len(os.listdir(os.path.join(scannet_path, scene, 'color')))
        for i in range(frames):
            if i % 20 == 0:
                color_file = os.path.join(scannet_path, scene, 'color', str(i) + '.jpg')
                depth_file = os.path.join(scannet_path, scene, 'depth', str(i) + '.png')
                label_file = os.path.join(scannet_path, scene, 'label-filt', str(i) + '.png')
                pose_file = os.path.join(scannet_path, scene, 'pose', str(i) + '.txt')
                color = imageio.imread(color_file)
                color = cv2.resize(color, (width, height), interpolation=cv2.INTER_NEAREST)
                imageio.imwrite(os.path.join(output_color_path, str(i) + '.jpg'), color)

                depth = imageio.imread(depth_file)
                depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
                imageio.imwrite(os.path.join(output_depth_path, str(i) + '.png'), depth)

                # label = imageio.imread(label_file)
                # label = transform.resize(np.array(label), [height, width], order=0, preserve_range=True)
                # label_map = util.read_label_mapping('/home/smy/scannet/scannetv2-labels.combined.tsv', label_from='id', label_to='nyu40id')
                # mapped_label = map_label_image(label, label_map)
                # imageio.imwrite(os.path.join(output_label_path, str(i) + '.png'), mapped_label)
                # shutil.copy(pose_file, os.path.join(output_pose_path, str(i) + '.txt'))


if __name__ == '__main__':
    main()
