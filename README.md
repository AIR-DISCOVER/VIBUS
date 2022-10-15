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
    open3d==0.13.0 \
    protobuf==3.20.0

pip install potpourri3d pymeshlab
```

## Testing

### Semantic Segmentation on ScanNet

You may specify the paths to datasets and checkpoints in `semantic_segmentation/scannet_ss_test.sh`

```shell
cd semantic_segmentation/
./scannet_ss_test.sh
```

### Semantic Segmentation on S3DIS

You may specify the paths to datasets and checkpoints in `semantic_segmentation/s3dis_ss_test.sh`

```shell
cd semantic_segmentation/
./s3dis_ss_test.sh
```

### Semantic Segmentation on Semantic3D

You may specify the paths to datasets and checkpoints in `semantic_segmentation/semantic3d_ss_test.sh`

```shell
cd semantic_segmentation/
./semantic3d_ss_test.sh
```

### Instance Segmentation on ScanNet

You may specify the paths to datasets and checkpoints in `instance_segmentation/scannet_is_test.sh`

```shell
cd instance_segmentation/
./scannet_is_test.sh
```

### Instance Segmentation on S3DIS

You may specify the paths to datasets and checkpoints in `instance_segmentation/s3dis_is_test.sh`

```shell
cd instance_segmentation/
./s3dis_is_test.sh
```

## Viewpoint-Bottleneck Pretraining (self supervised)

```shell
cd pretrain/
./run.sh
```

## Supervised Training / Fine-tuning

### Semantic Segmentation on ScanNet

You may specify the paths to the datasets in `semantic_segmentation/scannet_ss_train.sh`

```shell
cd semantic_segmentation/
./scannet_ss_train.sh
```

### Semantic Segmentation on S3DIS

You may specify the paths to the datasets in `semantic_segmentation/s3dis_ss_train.sh`

```shell
cd semantic_segmentation/
./s3dis_ss_train.sh
```

### Semantic Segmentation on Semantic3D

You may specify the paths to the datasets in `semantic_segmentation/semantic3d_ss_train.sh`

```shell
cd semantic_segmentation/
./semantic3d_ss_train.sh
```

### Instance Segmentation on ScanNet

You may specify the paths to the datasets in `instance_segmentation/scannet_is_train.sh`

```shell
cd instance_segmentation/
./scannet_is_train.sh
```

### Instance Segmentation on S3DIS

You may specify the paths to the datasets in `instance_segmentation/s3dis_is_train.sh`

```shell
cd instance_segmentation/
./s3dis_is_train.sh
```

## Model Zoo

### Viewpoint Bottleneck (VIB) Self-Supervised Pretrain

<div class="tg-wrap"><table style="undefined;table-layout: fixed; width: 492px">
<thead>
  <tr>
    <th>Dataset</th>
    <th>Task</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>ScanNet</td>
    <td><a href="https://drive.google.com/file/d/1oRIHlEu1fS2eKpaIyi1J7BCIkKSn174k/view?usp=sharing">Google Drive</a></td>
  </tr>
  <tr>
    <td>S3DIS</td>
    <td>Preparing</td>
  </tr>
  <tr>
    <td>Semantic3D</td>
    <td>Preparing</td>
  </tr>
</tbody>
</table></div>

### Final Checkpoints

<div class="tg-wrap"><table style="undefined;table-layout: fixed; width: 1079px">
<colgroup>
<col style="width: 245px">
<col style="width: 110px">
<col style="width: 169px">
<col style="width: 290px">
<col style="width: 265px">
</colgroup>
<thead>
  <tr>
    <th rowspan="2">Dataset</th>
    <th colspan="2" rowspan="2">Supervision</th>
    <th colspan="2">Task</th>
  </tr>
  <tr>
    <th>Semantic Segmentation</th>
    <th>Instance Segmentation</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="9">ScanNet</td>
    <td rowspan="4">Limited Annotations</td>
    <td>20 pts.</td>
    <td>Preparing</td>
    <td>Preparing</td>
  </tr>
  <tr>
    <td>50 pts.</td>
    <td>Preparing</td>
    <td>Preparing</td>
  </tr>
  <tr>
    <td>100 pts.</td>
    <td>Preparing</td>
    <td>Preparing</td>
  </tr>
  <tr>
    <td>200 pts.</td>
    <td>Preparing</td>
    <td>Preparing</td>
  </tr>
  <tr>
    <td rowspan="4">Limited Reconstructions</td>
    <td>1%</td>
    <td>Preparing</td>
    <td>Preparing</td>
  </tr>
  <tr>
    <td>5%</td>
    <td>Preparing</td>
    <td>Preparing</td>
  </tr>
  <tr>
    <td>10%</td>
    <td>Preparing</td>
    <td>Preparing</td>
  </tr>
  <tr>
    <td>20%</td>
    <td>Preparing</td>
    <td>Preparing</td>
  </tr>
  <tr>
    <td colspan="2">Full</td>
    <td>Preparing</td>
    <td>Preparing</td>
  </tr>
  <tr>
    <td rowspan="5">S3DIS</td>
    <td rowspan="4">Limited Annotations</td>
    <td>20 pts.</td>
    <td>Preparing</td>
    <td>Preparing</td>
  </tr>
  <tr>
    <td>50 pts.</td>
    <td>Preparing</td>
    <td>Preparing</td>
  </tr>
  <tr>
    <td>100 pts.</td>
    <td>Preparing</td>
    <td>Preparing</td>
  </tr>
  <tr>
    <td>200 pts.</td>
    <td>Preparing</td>
    <td>Preparing</td>
  </tr>
  <tr>
    <td colspan="2">Full</td>
    <td>Preparing</td>
    <td>Preparing</td>
  </tr>
  <tr>
    <td rowspan="5">Semantic3D</td>
    <td rowspan="4">Limited Annotations</td>
    <td>20 pts.</td>
    <td>Preparing</td>
    <td rowspan="5">N/A</td>
  </tr>
  <tr>
    <td>50 pts.</td>
    <td>Preparing</td>
  </tr>
  <tr>
    <td>100 pts.</td>
    <td>Preparing</td>
  </tr>
  <tr>
    <td>200 pts.</td>
    <td>Preparing</td>
  </tr>
  <tr>
    <td colspan="2">Full</td>
    <td>Preparing</td>
  </tr>
</tbody>
</table></div>