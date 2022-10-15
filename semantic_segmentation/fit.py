from argparse import ArgumentParser
import os
from plyfile import PlyData
import numpy as np
from sufield.fit import mixture_filter, BetaDistribution, GammaDistribution
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

def filter_uncertainty_from_obj(dataset_root: str, stat_root: str, save_root: str, cuda=True):
    import torch
    for plyfile in os.listdir(dataset_root):
        assert plyfile.endswith('.ply')
        plyfile_path = os.path.join(dataset_root, plyfile)
        plydata = PlyData.read(plyfile_path)
        (vertices, rgbs, _), faces = plydata_to_array(plydata)
        stat_path = os.path.join(stat_root, plyfile)
        predicted_labels = np.asarray(torch.load(stat_path[:-4] + '_predicted.obj'))
        uncertainty_scores = np.asarray(torch.load(stat_path[:-4] + '_unc.obj'))

        init_dist_a = GammaDistribution(2, 3)
        init_dist_b = GammaDistribution(3, 2)
        filter_mask = mixture_filter(uncertainty_scores, init_dist_a, init_dist_b, cuda=cuda)

        save_path = os.path.join(save_root, plyfile)
        new_labels = predicted_labels[filter_mask]
        construct_plydata(vertices, rgbs, new_labels).write(save_path)

def filter_uncertainty(dataset_root: str, save_root: str, uncertainty_stat_root: str):
    for plyfile in os.listdir(dataset_root):
        assert plyfile.endswith('.ply')
        assert os.path.isfile(os.path.join(uncertainty_stat_root, plyfile[:-4] + '.npy'))
        ...

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset_root", type=str)
    parser.add_argument("--stat_root", type=str)
    parser.add_argument("--save_root", type=str)
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--action", type=str)

    config = parser.parse_args()
    if config.action == "uncertainty":
        filter_uncertainty_from_obj(config.dataset_root, config.stat_root, config.save_root, config.use_cuda)
    elif config.action == "spectrum":
        filter_spectral(config.dataset_root, config.save_root)
    elif config.action == "joint":
        raise NotImplementedError
    else:
        raise NotImplementedError