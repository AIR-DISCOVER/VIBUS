import numpy as np
from lib.pc_utils import read_plyfile, save_point_cloud

f = '/home/aidrive1/workspace/luoly/dataset/Min_scan/scan_processed/train/scene0031_00.ply'

pointcloud = read_plyfile(f)

s = pointcloud.shape[0]
t = max(pointcloud[:,2])
pindex = np.random.randint(0, s, (8192,))


for k in range(s):
        pointcloud[k,4] = 255
        pointcloud[k,5] = 255
        pointcloud[k,6] = (pointcloud[k,2] / t)*255

point = pointcloud[pindex]
out1 = '/home/aidrive1/workspace/luoly/dataset/scene0031_00_14sampling.ply'
out2 = '/home/aidrive1/workspace/luoly/dataset/scene0031_00_14.ply'
save_point_cloud(point, out1, with_label=True, verbose=False)
save_point_cloud(pointcloud, out2, with_label=True, verbose=False)

#pointcloud[k,5] = ((t-pointcloud[k,1]) / t)*255