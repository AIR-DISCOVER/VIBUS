"""
spec_cluster.py
"""
from copy import deepcopy
from typing import Tuple

import numpy as np
import open3d as o3d
import potpourri3d as pp3d
import torch
from IPython import embed
from plyfile import PlyData
from sklearn.cluster import KMeans

from .downsample import downsample
from .utils import Timer, plydata_to_array
from .vis import dye_semantics


def geodesic_correlation_matrix(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    distances = []
    gd_solver = pp3d.MeshHeatMethodDistanceSolver(vertices, faces)
    for i in range(len(vertices)):
        distances.append(gd_solver.compute_distance(i))
    distances = np.array(distances)
    distances = distances / distances.mean()  # TODO check this lines
    return distances


def angular_correlation_matrix(vertices: np.ndarray, search_range=10) -> np.ndarray:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=search_range))
    normals = np.asarray(pcd.normals)
    distances = 1 - np.fabs(normals @ normals.T)
    distances = distances / distances.mean()  # TODO check this line
    return distances


def spectral_cluster(
    dist_mat: np.ndarray,
    init_indices: np.ndarray,
    temperature=None,
    embedding_length=20,
    cuda=True,
) -> Tuple[np.ndarray, np.ndarray]:
    if not cuda:
        dist_mat: np.ndarray = (dist_mat + dist_mat.T) / 2
        sigma = dist_mat.mean() if temperature is None else temperature
        affinity_mat = np.e**(-dist_mat / (2 * sigma**2))
        _coef = np.sqrt(np.diag(1 / affinity_mat.sum(axis=1)))
        normalized_affinity = _coef @ affinity_mat @ _coef
        eigen_vals, eigen_vecs = np.linalg.eigh(normalized_affinity)
        eigen_vals = eigen_vals[-embedding_length:][::-1]
        embeddings = eigen_vecs[:, -embedding_length:][:, ::-1]
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    else:
        with torch.no_grad():
            dist_mat: torch.tensor = torch.tensor((dist_mat + dist_mat.T) / 2, device='cuda')
            sigma = dist_mat.mean() if temperature is None else temperature
            affinity_mat = np.e**(-dist_mat / (2 * sigma**2))
            _coef = torch.sqrt(torch.diag(1 / affinity_mat.sum(axis=1)))
            normalized_affinity = _coef @ affinity_mat @ _coef
            eigen_vals, eigen_vecs = torch.linalg.eigh(normalized_affinity)
            embeddings = eigen_vecs[:, -embedding_length:].flip([1])
            embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
            embeddings = embeddings.cpu().numpy()

    cl_result = KMeans(n_clusters=len(init_indices), init=embeddings[init_indices]).fit(embeddings)
    cluster_indices = init_indices[cl_result.labels_]spec
    confidence = cl_result.transform(embeddings).min(axis=1)
    return cluster_indices, confidence


def test_spectral_cluster():
    import matplotlib.pyplot as plt

    points = np.random.rand(1000, 2)
    points[0][0] = 0
    points[0][1] = 0
    points[2][0] = 0
    points[2][1] = 1
    points[3][0] = 1
    points[3][1] = 0
    points[9][0] = 1
    points[9][1] = 1
    # embed()
    dist_mat = - 2 * points[:, 0].reshape(-1, 1) @ points[:, 0].reshape(1, -1) \
               - 2 * points[:, 1].reshape(-1, 1) @ points[:, 1].reshape(1, -1) \
               + (points[:, 0] ** 2).reshape(1, -1) + (points[:, 0] ** 2).reshape(-1, 1) \
               + (points[:, 1] ** 2).reshape(1, -1) + (points[:, 1] ** 2).reshape(-1, 1)
    print((dist_mat - dist_mat.T).sum())
    print(dist_mat[5][10] - np.linalg.norm(points[5] - points[10]))
    indices, confidence = spectral_cluster(dist_mat, np.array([0, 2, 3, 9]), embedding_length=5)

    plt.scatter([0], [0], c='r', s=50)
    plt.scatter([0], [1], c='g', s=50)
    plt.scatter([1], [0], c='y', s=50)
    plt.scatter([1], [1], c='b', s=50)
    plt.scatter(points[:, 0][indices == 0], points[:, 1][indices == 0], c='r')
    plt.scatter(points[:, 0][indices == 2], points[:, 1][indices == 2], c='g')
    plt.scatter(points[:, 0][indices == 3], points[:, 1][indices == 3], c='y')
    plt.scatter(points[:, 0][indices == 9], points[:, 1][indices == 9], c='b')
    plt.show()


if __name__ == '__main__':
    label_cnt = 50
    coef = 0.9
    timer = Timer()
    input_mesh_path = '/home/tb5zhh/Downloads/full_mesh.ply'
    downsampled_sparse_vis_path = '/home/tb5zhh/Downloads/full_ds_test.ply'
    downsampled_prediction_save_path = '/home/tb5zhh/Downloads/ds_predicted.ply'
    prediction_save_path = '/home/tb5zhh/Downloads/predicted.ply'

    full_mesh_path = input_mesh_path
    plydata = PlyData.read(full_mesh_path)
    (_, _, ground_truth_labels), _ = plydata_to_array(plydata)
    ds_plydata, mapping, inverse = downsample(plydata)
    init_indices = np.random.choice(len(ds_plydata['vertex']), 50, replace=False)

    # Save visualization of downsampled point clouds
    # along with random ground-truth labels
    ds_labels = deepcopy(ds_plydata['vertex']['label'])
    ds_plydata['vertex']['label'] = 255
    ds_plydata['vertex']['label'][init_indices] = ds_labels[init_indices]
    dye_semantics(ds_plydata).write(downsampled_sparse_vis_path)
    (coords, rgbs, labels), faces = plydata_to_array(ds_plydata)

    timer.tic()
    geod_mat = geodesic_correlation_matrix(coords, faces)
    timer.tic('Geodesic distances')
    ang_mat = angular_correlation_matrix(coords)
    timer.tic('Angular distances')
    dist_mat = coef * geod_mat + (1 - coef) * ang_mat
    indices, confidence = spectral_cluster(dist_mat, init_indices, embedding_length=40)
    timer.tic('Spectral clustering')

    embed()

    ds_plydata_predicted = deepcopy(ds_plydata)
    ds_plydata_predicted['vertex']['label'] = labels[indices]
    dye_semantics(ds_plydata_predicted).write(downsampled_prediction_save_path)

    plydata_predicted = deepcopy(plydata)
    plydata_predicted['vertex']['label'] = ds_plydata_predicted['vertex']['label'][inverse].flatten()
    dye_semantics(plydata_predicted).write(prediction_save_path)

    print((ground_truth_labels == plydata_predicted['vertex']['label']).sum() / len(ground_truth_labels))
