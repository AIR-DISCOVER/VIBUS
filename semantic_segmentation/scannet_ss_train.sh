#!/bin/bash
#SBATCH --job-name 20_fit_spec # 作业名为 example
#SBATCH --output job_log/20_fit_spec_%J.out    # 屏幕上的输出文件重定向到 [JOBID].out
#SBATCH --gres gpu:a100:1  # 使用 1 张 A100 显卡
#SBATCH --requeue
#SBATCH --time 4-0
NAME=20_fit_spec
# export CUDA_VISIBLE_DEVICES=1
echo $NAME
DATASET_PATH=/home/aidrive/tb5zhh/3d_scene_understand/SUField/results_0223/generate_datasets/${NAME}/train
TRAIN_BATCH_SIZE=22
LR=0.3
Scheduler=PolyLR
MODEL=Res16UNet34C
RUN_NAME=finetune_${NAME}_${LR}_${Scheduler}_${MODEL}
python -u new.py \
    --log_dir log \
    --seed 42 \
    --train_dataset ScannetVoxelization2cmDataset \
    --val_dataset ScannetVoxelization2cmtestDataset \
    --scannet_test_path /home/aidrive/tb5zhh/3d_scene_understand/data/full/train \
    --checkpoint_dir checkpoints \
    --num_workers 8 \
    --validate_step 100 \
    --optim_step 1 \
    --val_batch_size 8  \
    --save_epoch 5 \
    --max_iter 30000 \
    --scheduler $Scheduler \
    --do_train \
    --weights checkpoints_bf/pretrain_20000.pth\
    --run_name $RUN_NAME \
    --model $MODEL \
    --lr $LR \
    --train_batch_size $TRAIN_BATCH_SIZE  \
    --scannet_path $DATASET_PATH \
    --wandb True

    
