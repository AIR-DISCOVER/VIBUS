# VIBUS: Data-efficient 3D Scene Parsing with **VI**ewpoint **B**ottleneck and **U**ncertainty-**S**pectrum Modeling

> Beiwen Tian, Liyi Luo, Hao Zhao, Guyue Zhou

This repository contains implementation and checkpoints of *VIBUS: Data-efficient 3D Scene Parsing with **VI**ewpoint **B**ottleneck and **U**ncertainty-**S**pectrum Modeling*.

Our work has been accepted by **ISPRS Journal of Photogrammetry and Remote Sensing**.

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

cd SUField/
pip install -e . 
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

## Visualization

1. Collect the inference results

   Please change `SAVE_PATH` in `scannet_ss_test_collect_pred.sh`

   ```shell
   cd semantic_segmentation/
   ./scannet_ss_test_collect_pred.sh
   ```

2. Run a script so that the color of the point cloud is changed according to the predictions:
   
   ```shell
   cd semantic_segmentation/
   python visualize.py --dataset_root /save/path/in/step/1
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

## Perform Spectral / Uncertainty Filtering (on ScanNet)

### Spectral

1. Collect the inference results

   Please change `SAVE_PATH` in `scannet_ss_test_collect_pred.sh`

   ```shell
   cd semantic_segmentation/
   ./scannet_ss_test_collect_pred.sh
   ```

2. Perform Spectrum Filtering

   Please pass `SAVE_PATH` in step 1 as param for `--dataset_root`.
  
   ```shell
   cd semantic_segmentation/
   python fit.py --action spectrum --dataset_root /path/to/last/save/root --save_root /path/to/save/filtered/dataset
   ```

3. Use filtered dataset with pseudo labels to fine-tune model

   Please change `DATASET_PATH` to the save path for filtered dataset in step 2 in `scannet_ss_train.sh`.

   ```shell
   cd semantic_segmentation/
   ./scannet_ss_train.sh
   ```

### Uncertainty

1. Collect the inference results

   Please change `SAVE_PATH` in `scannet_ss_test_collect_pred_unc.sh`

   ```shell
   cd semantic_segmentation/
   ./scannet_ss_test_collect_pred_unc.sh
   ```

2. Perform Spectrum Filtering

   Please pass `SAVE_PATH` in step 1 as param for `--stat_root`.
  
   ```shell
   cd semantic_segmentation/
   python fit.py --action uncertainty --dataset_root /path/to/original/dataset --stat_root /path/to/last/save/root --save_root /path/to/save/filtered/dataset
   ```

3. Use filtered dataset with pseudo labels to fine-tune model

   Please change `DATASET_PATH` to the save path for filtered dataset in step 2 in `scannet_ss_train.sh`.

   ```shell
   cd semantic_segmentation/
   ./scannet_ss_train.sh
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
    <td><a href="https://drive.google.com/drive/folders/1drdTVnwfh6Qo7LMka2XWV71NzQ58TuaZ?usp=sharing">Google Drive</a></td>
  </tr>
  <tr>
    <td>Semantic3D</td>
    <td><a href="https://drive.google.com/drive/folders/1qWDbvsRzri7O4tpq69z9vhmJ65B2d2ba?usp=sharing">Google Drive</a></td>
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
    <td><a href="https://drive.google.com/file/d/1w7xViC62FYkJEJaYwQCBzBCuzP5mVJSZ/view?usp=sharing">Google Drive</a></td>
    <td><a href="https://drive.google.com/drive/folders/1mP6T1FjS3ueL7m4j_2GU72FQiqklQTGe?usp=sharing">Google Drive</a></td>
  </tr>
  <tr>
    <td>50 pts.</td>
    <td><a href="https://drive.google.com/file/d/1QpeXmjkxTytA_GDwTHFqaw-3aEltehWe/view?usp=sharing">Google Drive</a></td>
    <td><a href="https://drive.google.com/drive/folders/1mP6T1FjS3ueL7m4j_2GU72FQiqklQTGe?usp=sharing">Google Drive</a></td>
  </tr>
  <tr>
    <td>100 pts.</td>
    <td><a href="https://drive.google.com/file/d/1w5tgCAp1qyFIZWrZTC_PxSWh-ss2CUIk/view?usp=sharing">Google Drive</a></td>
    <td><a href="https://drive.google.com/drive/folders/1mP6T1FjS3ueL7m4j_2GU72FQiqklQTGe?usp=sharing">Google Drive</a></td>
  </tr>
  <tr>
    <td>200 pts.</td>
    <td><a href="https://drive.google.com/file/d/1RwoJrzVaSTqpAW8O5j5ou6cCSusVBvP1/view?usp=sharing">Google Drive</a></td>
    <td><a href="https://drive.google.com/drive/folders/1mP6T1FjS3ueL7m4j_2GU72FQiqklQTGe?usp=sharing">Google Drive</a></td>
  </tr>
  <tr>
    <td rowspan="4">Limited Reconstructions</td>
    <td>1%</td>
    <td><a href="https://drive.google.com/drive/folders/1mP6T1FjS3ueL7m4j_2GU72FQiqklQTGe?usp=sharing">Google Drive</a></td>
    <td><a href="https://drive.google.com/drive/folders/1mP6T1FjS3ueL7m4j_2GU72FQiqklQTGe?usp=sharing">Google Drive</a></td>
  </tr>
  <tr>
    <td>5%</td>
    <td><a href="https://drive.google.com/drive/folders/1mP6T1FjS3ueL7m4j_2GU72FQiqklQTGe?usp=sharing">Google Drive</a></td>
    <td><a href="https://drive.google.com/drive/folders/1mP6T1FjS3ueL7m4j_2GU72FQiqklQTGe?usp=sharing">Google Drive</a></td>
  </tr>
  <tr>
    <td>10%</td>
    <td><a href="https://drive.google.com/drive/folders/1mP6T1FjS3ueL7m4j_2GU72FQiqklQTGe?usp=sharing">Google Drive</a></td>
    <td><a href="https://drive.google.com/drive/folders/1mP6T1FjS3ueL7m4j_2GU72FQiqklQTGe?usp=sharing">Google Drive</a></td>
  </tr>
  <tr>
    <td>20%</td>
    <td><a href="https://drive.google.com/drive/folders/1mP6T1FjS3ueL7m4j_2GU72FQiqklQTGe?usp=sharing">Google Drive</a></td>
    <td><a href="https://drive.google.com/drive/folders/1mP6T1FjS3ueL7m4j_2GU72FQiqklQTGe?usp=sharing">Google Drive</a></td>
  </tr>
  <tr>
    <td colspan="2">Full</td>
    <td><a href="https://drive.google.com/drive/folders/1mP6T1FjS3ueL7m4j_2GU72FQiqklQTGe?usp=sharing">Google Drive</a></td>
    <td><a href="https://drive.google.com/drive/folders/1mP6T1FjS3ueL7m4j_2GU72FQiqklQTGe?usp=sharing">Google Drive</a></td>
  </tr>
  <tr>
    <td rowspan="5">S3DIS</td>
    <td rowspan="4">Limited Annotations</td>
    <td>20 pts.</td>
    <td><a href="https://drive.google.com/drive/folders/1drdTVnwfh6Qo7LMka2XWV71NzQ58TuaZ?usp=sharing">Google Drive</a></td>
    <td><a href="https://drive.google.com/drive/folders/1drdTVnwfh6Qo7LMka2XWV71NzQ58TuaZ?usp=sharing">Google Drive</a></td>
  </tr>
  <tr>
    <td>50 pts.</td>
    <td><a href="https://drive.google.com/drive/folders/1drdTVnwfh6Qo7LMka2XWV71NzQ58TuaZ?usp=sharing">Google Drive</a></td>
    <td><a href="https://drive.google.com/drive/folders/1drdTVnwfh6Qo7LMka2XWV71NzQ58TuaZ?usp=sharing">Google Drive</a></td>
  </tr>
  <tr>
    <td>100 pts.</td>
    <td><a href="https://drive.google.com/drive/folders/1drdTVnwfh6Qo7LMka2XWV71NzQ58TuaZ?usp=sharing">Google Drive</a></td>
    <td><a href="https://drive.google.com/drive/folders/1drdTVnwfh6Qo7LMka2XWV71NzQ58TuaZ?usp=sharing">Google Drive</a></td>
  </tr>
  <tr>
    <td>200 pts.</td>
    <td><a href="https://drive.google.com/drive/folders/1drdTVnwfh6Qo7LMka2XWV71NzQ58TuaZ?usp=sharing">Google Drive</a></td>
    <td><a href="https://drive.google.com/drive/folders/1drdTVnwfh6Qo7LMka2XWV71NzQ58TuaZ?usp=sharing">Google Drive</a></td>
  </tr>
  <tr>
    <td colspan="2">Full</td>
    <td><a href="https://drive.google.com/drive/folders/1drdTVnwfh6Qo7LMka2XWV71NzQ58TuaZ?usp=sharing">Google Drive</a></td>
    <td><a href="https://drive.google.com/drive/folders/1drdTVnwfh6Qo7LMka2XWV71NzQ58TuaZ?usp=sharing">Google Drive</a></td>
  </tr>
  <tr>
    <td rowspan="5">Semantic3D</td>
    <td rowspan="4">Limited Annotations</td>
    <td>20 pts.</td>
    <td><a href="https://drive.google.com/drive/folders/1qWDbvsRzri7O4tpq69z9vhmJ65B2d2ba?usp=sharing">Google Drive</a></td>
    <td rowspan="5">N/A</td>
  </tr>
  <tr>
    <td>50 pts.</td>
    <td><a href="https://drive.google.com/drive/folders/1qWDbvsRzri7O4tpq69z9vhmJ65B2d2ba?usp=sharing">Google Drive</a></td>
  </tr>
  <tr>
    <td>100 pts.</td>
    <td><a href="https://drive.google.com/drive/folders/1qWDbvsRzri7O4tpq69z9vhmJ65B2d2ba?usp=sharing">Google Drive</a></td>
  </tr>
  <tr>
    <td>200 pts.</td>
    <td><a href="https://drive.google.com/drive/folders/1qWDbvsRzri7O4tpq69z9vhmJ65B2d2ba?usp=sharing">Google Drive</a></td>
  </tr>
  <tr>
    <td colspan="2">Full</td>
    <td><a href="https://drive.google.com/drive/folders/1qWDbvsRzri7O4tpq69z9vhmJ65B2d2ba?usp=sharing">Google Drive</a></td>
  </tr>
</tbody>
</table></div>
