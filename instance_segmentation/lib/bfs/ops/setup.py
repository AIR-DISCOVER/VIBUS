from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

import os
_ext_src_root = os.path.abspath(os.environ['CONDA_PREFIX'])
setup(
    name='PG_OP',
    ext_modules=[
        CUDAExtension('PG_OP', [
            'src/bfs_cluster.cpp',
            'src/bfs_cluster_kernel.cu',
        ],
        extra_compile_args={
                "cxx": ["-O2", "-I{}".format("{}/include".format(_ext_src_root))],
                "nvcc": ["-O2", "-I{}".format("{}/include".format(_ext_src_root))],
            },)
    ],
    cmdclass={'build_ext': BuildExtension}
)