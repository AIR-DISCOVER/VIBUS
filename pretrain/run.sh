#!/bin/bash

set -exo pipefail

PRETRAIN_PATH=${PRETRAIN_PATH:-'None'}
DATASET_PATH=${DATASET_PATH:-'/data/luoly/dataset/Min_scan'}
LOG_DIR=${LOG_DIR:-'logs'}
SHOTS=${SHOTS:-200}
FEATURE_DIM=${FEATURE_DIM:-256}
BATCH_SIZE=${BATCH_SIZE:-1}
MODEL=${MODEL:-'Res16UNet34C'}

# TODO specify 
DATAPATH=$DATASET_PATH/scan_processed/train
TESTDATAPATH=$DATASET_PATH/scan_processed/train
TIME=$(date +"%Y-%m-%d_%H-%M-%S")
mkdir -p $LOG_DIR
LOG="$LOG_DIR/$TIME.txt"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-'5'}


# TODO remove distributed_world_size argument
python -m ddp_main \
    --feature_dim=$FEATURE_DIM \
    --distributed_world_size=1 \
    --train_phase=train \
    --is_train=True \
    --lenient_weight_loading=True \
    --stat_freq=1 \
    --val_freq=500 \
    --save_freq=500 \
    --model=${MODEL} \
    --conv1_kernel_size=5 \
    --normalize_color=True \
    --dataset=ScannetVoxelization2cmDataset \
    --testdataset=ScannetVoxelization2cmtestDataset \
    --batch_size=$BATCH_SIZE \
    --num_workers=1 \
    --num_val_workers=1 \
    --scannet_path=${DATAPATH} \
    --scannet_test_path=${TESTDATAPATH} \
    --return_transformation=False \
    --test_original_pointcloud=False \
    --save_prediction=False \
    --lr=0.1 \
    --scheduler=PolyLR \
    --max_iter=30000 \
    --log_dir=${LOG_DIR} \
    --weights=${PRETRAIN_PATH} \
     2>&1 | tee -a "$LOG"
