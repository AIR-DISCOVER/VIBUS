#!/bin/bash
#SBATCH --job-name pre_S3_is_100_12 # 作业名为 example
#SBATCH --output job_log/pre_S3_is_100_12_%J.out    # 屏幕上的输出文件重定向到 [JOBID].out
#SBATCH --gres gpu:a100:1  # 使用 1 张 A100 显卡
#SBATCH --requeue
#SBATCH --time 3-0
NAME=pre_S3_is_100_12

echo $NAME
DATASET_PATH=/DATA_EDS/luoly/datasets/S3DIS/Stanford_preprocessing/instance/100
TRAIN_BATCH_SIZE=20
LR=0.1
MODEL=Res16UNet34C
RUN_NAME=finetune_${NAME}_${LR}_${MODEL}
python -u new.py \
    --log_dir log \
    --seed 42 \
    --train_dataset StanfordArea5Dataset \
    --val_dataset StanfordArea5testDataset \
    --stanford3d_test_path /DATA_EDS/luoly/datasets/S3DIS/Stanford_preprocessing/instance/full \
    --checkpoint_dir checkpoints \
    --num_workers 4 \
    --num_classes 13 \
    --validate_step 100 \
    --optim_step 1 \
    --val_batch_size 1  \
    --save_epoch 5 \
    --max_iter 20000 \
    --scheduler PolyLR \
    --do_train \
    --run_name $RUN_NAME \
    --weights /DATA_EDS/luoly/code/S3DIS_instance/pretrain/logs/new_pre_is_2W.pth \
    --model $MODEL \
    --lr $LR \
    --train_batch_size $TRAIN_BATCH_SIZE  \
    --stanford3d_path $DATASET_PATH \
    --wandb False 

    