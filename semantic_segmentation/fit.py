import os
from plyfile import PlyData
import numpy as np
from sufield.fit import mixture_filter, BetaDistribution
from sufield.spec_cluster import geodesic_correlation_matrix, angular_correlation_matrix, spectral_cluster
from sufield.utils import plydata_to_array, construct_plydata

def filter_spectral(dataset_root: str, save_root: str, center_count=20, cuda=True, ratio=0.3):
    for plyfile in os.listdir(dataset_root):
        assert plyfile.endswith('.ply')
        plyfile_path = os.path.join(dataset_root, plyfile)
        plydata = PlyData.read(plyfile_path)
        (vertices, rgbs, labels), faces = plydata_to_array(plydata)
        geod_mat = geodesic_correlation_matrix(vertices, faces)
        ang_mat = angular_correlation_matrix(vertices)
        dist_mat = ratio * ang_mat + (1 - ratio) * geod_mat
        init_indices = np.random.choice(len(vertices), center_count, replace=False)
        cluster_indices, confidence = spectral_cluster(dist_mat, init_indices, cuda=cuda)

        confidence = (confidence - confidence.min()) / (confidence.max() - confidence.min())
        init_dist_a = BetaDistribution(2, 10)
        init_dist_b = BetaDistribution(10, 2)
        filter_mask = mixture_filter(confidence, init_dist_a, init_dist_b, step=500)

        save_path = os.path.join(save_root, plyfile)
        new_labels = labels[cluster_indices][filter_mask]
        construct_plydata(vertices, rgbs, new_labels).write(save_path)

def filter_uncertainty(dataset_root: str, save_root: str, uncertainty_stat_root: str):
    for plyfile in os.listdir(dataset_root):
        assert plyfile.endswith('.ply')
        assert os.path.isfile(os.path.join(uncertainty_stat_root, plyfile[:-4] + '.npy'))
        ...