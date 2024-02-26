import os
import open3d as o3d
import numpy as np
from os.path import join


def gen_viewpoints_from_cluster(mesh_simp, cluster_labels, cluster_id):
    v_coords = np.asarray(mesh_simp.vertices)
    v_norms = np.asarray(mesh_simp.vertex_normals)
    inds = np.where(cluster_labels == cluster_id)
    this_coords = v_coords[inds]
    this_norms = v_norms[inds]
    center = np.mean(this_coords)
    norm = np.mean(this_norms)


if __name__ == '__main__':
    scans_path = ''