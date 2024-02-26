import shutil
from os.path import join
import os, shutil
scans_path = '/mnt/disk/scannet3D/2D'
for scene in sorted(os.listdir(scans_path)):
    viewpoint_path = join(scans_path, scene, 'viewpoint')
    shutil.rmtree(viewpoint_path)
