from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
from plyfile import PlyData, PlyElement


class Timer:

    def __init__(self, quiet=False):
        self.now = datetime.now()
        self.start = datetime.now()
        self.quiet = quiet

    def tic(self, event: Optional[str] = None):
        now = datetime.now()
        if not self.queit and event is not None:
            print(f'============={event}===============')
            print('Delta: ' + str((now - self.now).seconds) + 's ')
            print('Total: ' + str((now - self.start).seconds) + 's ')
        self.now = now


def plydata_to_array(
    plydata,
    coords=['x', 'y', 'z'],
    features=['red', 'green', 'blue'],
    label=['label'],
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    TODO add docstring
    """
    component_names = [i.name for i in plydata.elements]
    assert 'vertex' in component_names

    def parse(fields):
        if fields is not None:
            if len(fields) > 1:
                ret = np.stack([plydata['vertex'][field] for field in fields], axis=-1)
            else:
                ret = np.array(plydata['vertex'][fields[0]])
        else:
            ret = None
        return ret

    ret_coords = parse(coords)
    ret_feats = parse(features)
    ret_labels = parse(label)
    if 'face' in component_names:
        ret_faces = np.concatenate(plydata['face']['vertex_indices']).reshape(-1, 3)
    return (ret_coords, ret_feats, ret_labels), ret_faces


def add_fields_online(
    plydata: PlyData,
    add_fields=[('label', 'u1')],
    add_vals=[None],
) -> PlyData:
    p = plydata
    v, f = p.elements
    a = np.empty(len(v.data), v.data.dtype.descr + add_fields)
    for name in v.data.dtype.fields:
        a[name] = v[name]
    if add_vals is not None:
        for (name, _), val in zip(add_fields, add_vals):
            if val is not None:
                a[name] = val
    v = PlyElement.describe(a, 'vertex')
    p = PlyData([v, f], text=True)
    return p


def construct_plydata(v_coords: np.ndarray, v_rgbs: np.ndarray = None, v_labels: np.ndarray = None) -> PlyData:
    plydata_t = [
        ('x', 'f4'),
        ('y', 'f4'),
        ('z', 'f4'),
        ('red', 'u1'),
        ('green', 'u1'),
        ('blue', 'u1'),
        ('alpha', 'u1'),
        ('label', 'u1'),
    ]
    plydata = np.zeros(len(v_coords), dtype=plydata_t)
    plydata['x'] = v_coords[:, 0]
    plydata['y'] = v_coords[:, 1]
    plydata['z'] = v_coords[:, 2]
    if v_rgbs is not None:
        plydata['red'] = v_rgbs[:, 0]
        plydata['green'] = v_rgbs[:, 1]
        plydata['blue'] = v_rgbs[:, 2]
        if v_rgbs.shape[1] > 3:
            plydata['alpha'] = v_rgbs[:, 3]
        else:
            plydata['alpha'] = 255
    if v_labels is not None:
        plydata['label'] = v_labels
    return plydata
