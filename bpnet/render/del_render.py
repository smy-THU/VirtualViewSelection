import shutil
from os.path import join
import os, shutil
scans_path = '/mnt/disk/scannet3D/2D'
for scene in sorted(os.listdir(scans_path)):
    c_path = join(scans_path, scene, 'render_color')
    d_path = join(scans_path, scene, 'render_depth')
    shutil.rmtree(c_path)
    shutil.rmtree(d_path)
