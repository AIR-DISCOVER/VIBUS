# VIBUS: Data-efficient 3D Scene Parsing with **VI**ewpoint **B**ottleneck and **U**ncertainty-**S**pectrum Modeling

> Beiwen Tian, Liyi Luo, Hao Zhao, Guyue Zhou

This repository contains implementation and checkpoints of *VIBUS: Data-efficient 3D Scene Parsing with **VI**ewpoint **B**ottleneck and **U**ncertainty-**S**pectrum Modeling*.

<!-- ## Citation -->

## Prepare Conda environment

The version of CUDA-Toolkit should **NOT** be higher than 11.1.

```shell
# Create conda environment
conda create -n vibus python=3.8
conda activate vibus

# Install MinkowskiEngine
export CUDA_HOME=/usr/local/cuda-11.1
conda install openblas-devel -c anaconda
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 \
    -f https://download.pytorch.org/whl/torch_stable.html
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps \
    --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" \
    --install-option="--blas=openblas"

# Install pointnet2 package
cd pointnet2
python setup.py install

# Install bfs package
conda install -c bioconda google-sparsehash
cd instanc_segmentation/lib/bfs/ops
python setup.py build_ext --include-dirs=${CONDA_PREFIX}/include
python setup.py install

# Install other requirements
pip install \
    easydict==1.9 \
    imageio==2.9.0 \
    plyfile==0.7.4 \
    tensorboardx==2.2 \
    open3d==0.13.0
```



## Model Zoo