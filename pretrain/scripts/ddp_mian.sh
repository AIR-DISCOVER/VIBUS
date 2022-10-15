#!/bin/bash

set -x
# Exit script when a command returns nonzero state
set -e

set -o pipefail
export CUDA_VISIBLE_DEVICES='0,4,5'
PYTHONUNBUFFERED="True"
DATAPATH=/home/aidrive1/workspace/luoly/dataset/Min_scan/one
 # Download ScanNet segmentation dataset and change the path here
PRETRAIN="none" # For finetuning, use the checkpoint path here.
MODEL=Res16UNet34C
BATCH_SIZE=${BATCH_SIZE:-16}
TIME=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_DIR=./tmp_dir_scannet
mkdir -p $LOG_DIR
LOG="$LOG_DIR/$TIME.txt"

python -m ddp_main \
    --train_phase=train \
    --is_train=True \
    --lenient_weight_loading=True \
    --stat_freq=1 \
    --val_freq=500 \
    --save_freq=500 \
    --model=${MODEL} \
    --conv1_kernel_size=3 \
    --normalize_color=True \
    --dataset=ScannetVoxelization2cmDataset \
    --batch_size=$BATCH_SIZE \
    --num_workers=1 \
    --num_val_workers=1 \
    --scannet_path=${DATAPATH} \
    --return_transformation=False \
    --test_original_pointcloud=False \
    --save_prediction=False \
    --lr=0.8 \
    --scheduler=PolyLR \
    --max_iter=60000 \
    --log_dir=${LOG_DIR} \
    --weights=${PRETRAIN} \
    $3 2>&1 | tee -a "$LOG"
