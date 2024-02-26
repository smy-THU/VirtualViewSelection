import imageio
import numpy as np
depth_file1 = '/mnt/disk/scannet3D/2D/scene0000_00/depth/0.png'
depth_file2 = '/mnt/disk/scannet3D/2D/scene0000_00/render_depth/0.png'

a = np.asarray(imageio.imread(depth_file1))
b = np.asarray(imageio.imread(depth_file2))
print(a,b)