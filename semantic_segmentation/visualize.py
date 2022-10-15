from argparse import ArgumentParser
import os
from pickletools import string1
from plyfile import PlyData
from sufield.vis import dye_semantics

def visualize_save(dataset_root: string1):
    for plyfile in os.listdir(dataset_root):
        assert plyfile.endswith('.ply')
        plyfile_path = os.path.join(dataset_root, plyfile)
        save_path = os.path.join(dataset_root, plyfile[:-4] + '_vis.ply')
        plydata = PlyData.read(plyfile_path)
        dye_semantics(plydata).write(save_path)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_root", type=str)
    config = parser.parse_args()
    visualize_save(config.dataset_root)
