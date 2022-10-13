#!/bin/bash
#SBATCH --job-name pre_inseg_20 # 作业名为 example
#SBATCH --output job_log/pre_inseg_20_%J.out    # 屏幕上的输出文件重定向到 [JOBID].out
#SBATCH --gres gpu:a100:1  # 使用 1 张 A100 显卡
#SBATCH --requeue
#SBATCH --time 5-0
NAME=pre_lr_is_20
echo $NAME

WEIGHTS=/DATA_EDS/luoly/code_orginal/scannet_instance/pretrain/logs/pre_is_scannet_2W.pth
DATASET_PATH=/DATA_EDS/luoly/datasets/Scannet/instance/full/train
TEST_DATASET_PATH=/DATA_EDS/luoly/datasets/Scannet/instance/full/train

TRAIN_BATCH_SIZE=2
LR=0.1
MODEL=Res16UNet34C
RUN_NAME=finetune_${NAME}_${LR}_${MODEL}

python -u new.py \
    --log_dir log \
    --seed 42 \
    --train_dataset ScannetVoxelization2cmDataset \
    --val_dataset ScannetVoxelization2cmtestDataset \
    --checkpoint_dir checkpoints \
    --num_workers 8 \
    --num_classes 20 \
    --validate_step 100 \
    --optim_step 1 \
    --val_batch_size 1 \
    --save_epoch 5 \
    --max_iter 30000 \
    --scheduler PolyLR \
    --do_validate \
    --run_name $RUN_NAME \
    --model $MODEL \
    --weights $WEIGHTS \
    --lr $LR \
    --train_batch_size $TRAIN_BATCH_SIZE  \
    --scannet_path $DATASET_PATH \
    --scannet_test_path $TEST_DATASET_PATH \
    --wandb False