import open3d as o3d
import numpy as np
from scripts.vis_results import create_color_palette
from sklearn.cluster import KMeans, DBSCAN
from sklearn import metrics
NORMAL_WEIGHT = 20
certainty_threshold = 0.9


def cal_normal(mesh, certainty):
    inds_convert = np.where(certainty < certainty_threshold)[0]
    print(inds_convert.shape)
    mesh.compute_vertex_normals()
    v_coords = np.array(mesh.vertices)
    v_normals = np.array(mesh.vertex_normals)
    v_colors = np.asarray(mesh.vertex_colors)
    v_normals *= NORMAL_WEIGHT
    v_feats = np.concatenate((v_normals, v_coords, v_colors), axis=1)
    v_feats = v_feats[inds_convert]
    kmeans_model = KMeans(n_clusters=40).fit(v_feats)
    labels = kmeans_model.labels_

    # dbscan_model = DBSCAN(eps=2.5).fit(v_feats)
    # labels = dbscan_model.labels_
    metrics.silhouette_score(v_feats, labels)
    color_palette = create_color_palette()
    for i in range(len(labels)):
        v_colors[inds_convert[i]] = color_palette[labels[i]] / 255.0
    o3d.visualization.draw_geometries([mesh])


if __name__ == '__main__':
    ply_file = '/mnt/disk/scannet3D/val/scene0011_00_vh_clean_2.ply'
    certainty_file = 'result/origin/scene0011_00certainty.txt'
    mesh = o3d.io.read_triangle_mesh(ply_file)
    certainty_list = []
    with open(certainty_file) as f:
        for c in f:
            certainty_list.append(float(c))
    certainty = np.asarray(certainty_list)
    # o3d.visualization.draw_geometries([mesh])
    print(len(mesh.vertices))
    voxel_size = 0.05
    mesh_simp = mesh.simplify_vertex_clustering(voxel_size=voxel_size, contraction=o3d.geometry.SimplificationContraction.Average)
    # o3d.visualization.draw_geometries([mesh])
    print(len(mesh_simp.vertices))
    vertices = np.asarray(mesh.vertices)
    vertices_simp = np.asarray(mesh_simp.vertices)



    inds_convert0 = np.where((vertices==vertices_simp[0][0]))
    cal_normal(mesh_simp, certainty)
