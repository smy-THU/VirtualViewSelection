import sys
import open3d as o3d
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn import metrics
from os.path import join
import os
from tqdm import tqdm
sys.path.append('../')
from scripts.vis_results import create_color_palette

NORMAL_WEIGHT = 2
certainty_threshold = 0.9


def clustering(scene):
    scans_path = '/home/smy/scannet/scans'
    myscans_path = '/mnt/disk/scannet3D/2D'
    ply_file = join(scans_path, scene, scene + '_vh_clean_2.ply')
    mesh = o3d.io.read_triangle_mesh(ply_file)
    mesh = mesh.simplify_vertex_clustering(voxel_size=0.05,
                                           contraction=o3d.geometry.SimplificationContraction.Average)
    mesh.compute_vertex_normals()
    out_file = join(myscans_path, scene, scene + '_simplified.ply')
    o3d.io.write_triangle_mesh(out_file, mesh)

    v_coords = np.array(mesh.vertices)
    v_normals = np.array(mesh.vertex_normals)
    v_colors = np.asarray(mesh.vertex_colors)
    v_normals *= NORMAL_WEIGHT
    v_feats = np.concatenate((v_normals, v_coords, v_colors), axis=1)
    coords_max = mesh.get_max_bound()
    coords_min = mesh.get_min_bound()
    n_clusters = int((coords_max[0] - coords_min[0]) * (coords_max[1] - coords_min[1]) / 4) + 5
    n_clusters = min(41, n_clusters)
    kmeans_model = KMeans(n_clusters=n_clusters).fit(v_feats)
    labels = kmeans_model.labels_
    labels_file = join(myscans_path, scene, scene + '_labels_simp.txt')
    labels_ply_file = join(myscans_path, scene, scene + '_simplified_clustering.ply')
    np.savetxt(labels_file, labels, fmt='%d')
    # metrics.silhouette_score(v_feats, labels)
    color_palette = create_color_palette()
    for i in range(len(labels)):
        v_colors[i] = color_palette[labels[i]] / 255.0
    o3d.io.write_triangle_mesh(labels_ply_file, mesh)
    # o3d.visualization.draw_geometries([mesh])


if __name__ == '__main__':
    scans_path = '/home/smy/scannet/scans'
    for i, scene in enumerate(tqdm(sorted(os.listdir(scans_path)))):
        clustering(scene)
