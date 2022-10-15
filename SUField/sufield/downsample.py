import tempfile

import open3d as o3d
import pymeshlab as ml
from plyfile import PlyData

from .utils import add_fields_online, plydata_to_array
from .vis import dye_semantics


def downsample(plydata: PlyData, times=3) -> PlyData:
    (coords, colors, labels), _ = plydata_to_array(plydata)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    f = tempfile.NamedTemporaryFile('w+b', suffix='.ply')
    plydata.write(f)
    f.seek(0)
    ms = ml.MeshSet()
    ms.load_new_mesh(f.name)
    f.close()

    for _ in range(times):
        ms.apply_filter('meshing_decimation_quadric_edge_collapse')
    ms.apply_filter('meshing_remove_unreferenced_vertices')
    f = tempfile.NamedTemporaryFile('w+b', suffix='.ply')
    ms.save_current_mesh(f.name)
    ds_plydata = PlyData.read(f)
    f.close()

    ds_plydata = add_fields_online(ds_plydata)
    (ds_coords, _, _), _ = plydata_to_array(ds_plydata)

    mapping = []
    for i in range(len(ds_plydata['vertex'])):
        [_, idx, _] = pcd_tree.search_knn_vector_3d(ds_coords[i], 1)
        mapping.append(idx)
        ds_plydata['vertex']['label'][i] = plydata['vertex']['label'][idx]

    ds_pcd = o3d.geometry.PointCloud()
    ds_pcd.points = o3d.utility.Vector3dVector(ds_coords)
    ds_pcd_tree = o3d.geometry.KDTreeFlann(ds_pcd)
    inverse = []
    for i in range(len(plydata['vertex'])):
        [_, idx, _] = ds_pcd_tree.search_knn_vector_3d(coords[i], 1)
        inverse.append(idx)
    return ds_plydata, mapping, inverse


def test_downsample():
    plydata = PlyData.read('/home/tb5zhh/Downloads/test.ply')
    dye_semantics(plydata).write('/home/tb5zhh/Downloads/test_mask.ply')
    ds_plydata, _, _ = downsample(plydata)
    ds_plydata.write('/home/tb5zhh/Downloads/testds.ply')
    dye_semantics(ds_plydata).write('/home/tb5zhh/Downloads/test_mask_ds.ply')
