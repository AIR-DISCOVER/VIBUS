#!/bin/bash
#SBATCH --job-name 20_fit_spec # 作业名为 example
#SBATCH --output job_log/20_fit_spec_%J.out    # 屏幕上的输出文件重定向到 [JOBID].out
#SBATCH --gres gpu:a100:1  # 使用 1 张 A100 显卡
#SBATCH --requeue
#SBATCH --time 4-0
NAME=20_fit_spec

WEIGHTS=${WEIGHTS:-/DATA_EDS/tb5zhh/legacy/3d_scene_understand/new_3dseg/stsegmentation/log/20_fit_unc_fixed/checkpoint_NoneRes16UNet34C.pth}
echo $WEIGHTS
# WEIGHTS=/DATA_EDS/tb5zhh/legacy/3d_scene_understand/3DScanSeg/checkpoints/pretrain_20000.pth
DATASET_PATH=/DATA_EDS/tb5zhh/legacy/3d_scene_understand/SUField/results_0223/generate_datasets/$NAME/train
TEST_DATASET_PATH=/DATA_EDS/tb5zhh/legacy/3d_scene_understand/data/full/train

TRAIN_BATCH_SIZE=8
LR=0.3
MODEL=Res16UNet34C
RUN_NAME=finetune_${NAME}_${LR}_${Scheduler}_${MODEL}
SAVE_DIR=/tmp

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
    --val_batch_size 8  \
    --save_epoch 5 \
    --max_iter 30000 \
    --scheduler PolyLR \
    --do_validate \
    --weights $WEIGHTS \
    --run_name $RUN_NAME \
    --model $MODEL \
    --lr $LR \
    --train_batch_size $TRAIN_BATCH_SIZE  \
    --scannet_path $DATASET_PATH \
    --scannet_test_path $TEST_DATASET_PATH \
    --save_prediction \
    --eval_result_dir $SAVE_DIR \
    --wandb False
