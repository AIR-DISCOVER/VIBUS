#!/bin/bash
#SBATCH --job-name new_s3d_ss_20_6k   # 作业名为 example
#SBATCH --output job_log/new_s3d_ss_20_6k_%J.out    # 屏幕上的输出文件重定向到 [JOBID].out
#SBATCH --gres gpu:a100-80G:1  # 使用 1 张 A100 显卡
#SBATCH --requeue
#SBATCH --time 5-0
NAME=new_s3d_ss_20_6k

echo $NAME
DATASET_PATH=/home/aidrive/luoly/datasets/semantic3d_preprocess/20
TRAIN_BATCH_SIZE=1
LR=0.1
MODEL=Res16UNet34C
RUN_NAME=finetune_${NAME}_${LR}_${MODEL}
python -u new.py \
    --log_dir log \
    --seed 42 \
    --train_dataset SemanticVoxelizationDataset \
    --val_dataset SemanticVoxelizationtestDataset \
    --semantic3d_test_path /home/aidrive/luoly/datasets/semantic3d_preprocess \
    --checkpoint_dir checkpoints \
    --num_workers 8 \
    --validate_step 50 \
    --optim_step 1 \
    --val_batch_size 1  \
    --save_epoch 5 \
    --max_iter 6000 \
    --scheduler PolyLR \
    --do_train \
    --run_name $RUN_NAME \
    --model $MODEL \
    --lr $LR \
    --train_batch_size $TRAIN_BATCH_SIZE  \
    --semantic3d_path $DATASET_PATH \
    --wandb True
