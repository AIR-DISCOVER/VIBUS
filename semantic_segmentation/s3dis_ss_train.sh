#!/bin/bash
#SBATCH --job-name pre_new_S3_ss_20_2W # 作业名为 example
#SBATCH --output job_log/pre_new_S3_ss_20_2W_%J.out    # 屏幕上的输出文件重定向到 [JOBID].out
#SBATCH --gres gpu:a100-80G:1  # 使用 1 张 A100 显卡
#SBATCH --requeue
#SBATCH --time 4-0
NAME=pre_new_S3_ss_20_2W

echo $NAME
DATASET_PATH=/home/aidrive/luoly/datasets/S3DIS/Stanford_preprocessing/semantic/20
TRAIN_BATCH_SIZE=20
LR=0.1
Scheduler=PolyLR
MODEL=Res16UNet34C
RUN_NAME=finetune_${NAME}_${LR}_${Scheduler}_${MODEL}
python -u new.py \
    --log_dir log \
    --seed 42 \
    --train_dataset StanfordArea5Dataset \
    --val_dataset StanfordArea5testDataset \
    --stanford3d_test_path /home/aidrive/luoly/datasets/S3DIS/Stanford_preprocessing/semantic/full \
    --checkpoint_dir checkpoints \
    --num_workers 8 \
    --validate_step 100 \
    --optim_step 1 \
    --val_batch_size 8  \
    --save_epoch 10 \
    --max_iter 20000 \
    --scheduler $Scheduler \
    --do_train \
    --run_name $RUN_NAME \
    --model $MODEL \
    --weights /home/aidrive/luoly/code/S3DIS_semantic/pretrain/log/2022-06-22.22:45:21/pre_ss_2W.pth \
    --lr $LR \
    --train_batch_size $TRAIN_BATCH_SIZE  \
    --stanford3d_path $DATASET_PATH \
    --wandb True


