import argparse


def str2opt(arg):
    assert arg in ['SGD', 'Adam']
    return arg


def str2scheduler(arg):
    assert arg in ['StepLR', 'PolyLR', 'ExpLR', 'SquaredLR']
    return arg


def str2bool(v):
    return v.lower() in ('true', '1')


def str2list(l):
    return [int(i) for i in l.split(',')]


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


arg_lists = []
parser = argparse.ArgumentParser()

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--model', type=str, default='Res16UNet34C', help='Model name')
net_arg.add_argument('--conv1_kernel_size', type=int, default=5, help='First layer conv kernel size')
net_arg.add_argument('--weights', type=str, default=None, help='Saved weights to load')
net_arg.add_argument('--weights_for_inner_model', type=str2bool, default=False, help='Weights for model inside a wrapper')
net_arg.add_argument('--dilations', type=str2list, default='1,1,1,1', help='Dilations used for ResNet or DenseNet')
net_arg.add_argument('--num_classes', type=int, default=20, help='Number of classes in the dataset')

# Wrappers
net_arg.add_argument('--wrapper_type', default='None', type=str, help='Wrapper on the network')
net_arg.add_argument('--wrapper_region_type', default=1, type=int, help='Wrapper connection types 0: hypercube, 1: HYPER_CROSS, (default: 1)')
net_arg.add_argument('--wrapper_kernel_size', default=3, type=int, help='Wrapper kernel size')
net_arg.add_argument('--wrapper_lr', default=1e-1, type=float, help='Used for freezing or using small lr for the base model, freeze if negative')

# Meanfield arguments
net_arg.add_argument('--meanfield_iterations', type=int, default=10, help='Number of meanfield iterations')
net_arg.add_argument('--crf_spatial_sigma', default=1, type=int, help='Trilateral spatial sigma')
net_arg.add_argument('--crf_chromatic_sigma', default=12, type=int, help='Trilateral chromatic sigma')

# Optimizer arguments
opt_arg = add_argument_group('Optimizer')
opt_arg.add_argument('--optimizer', type=str, default='SGD')
opt_arg.add_argument('--lr', type=float, default=1e-1)
opt_arg.add_argument('--sgd_momentum', type=float, default=0.9)
opt_arg.add_argument('--sgd_dampening', type=float, default=0.1)
opt_arg.add_argument('--adam_beta1', type=float, default=0.9)
opt_arg.add_argument('--adam_beta2', type=float, default=0.999)
opt_arg.add_argument('--weight_decay', type=float, default=1e-4)
opt_arg.add_argument('--param_histogram_freq', type=int, default=100)
opt_arg.add_argument('--save_param_histogram', type=str2bool, default=False)
opt_arg.add_argument('--iter_size', type=int, default=1, help='accumulate gradient')
opt_arg.add_argument('--bn_momentum', type=float, default=0.02)

# Scheduler
opt_arg.add_argument('--scheduler', type=str2scheduler, default='StepLR')
opt_arg.add_argument('--max_iter', type=int, default=2000)
opt_arg.add_argument('--step_size', type=int, default=2e4)
opt_arg.add_argument('--step_gamma', type=float, default=0.1)
opt_arg.add_argument('--poly_power', type=float, default=0.9)
opt_arg.add_argument('--exp_gamma', type=float, default=0.95)
opt_arg.add_argument('--exp_step_size', type=float, default=445)

# Directories
dir_arg = add_argument_group('Directories')
dir_arg.add_argument('--log_dir', type=str, default='outputs/default')
dir_arg.add_argument('--data_dir', type=str, default='data')

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--dataset', type=str, default='ScannetVoxelization2cmDataset')
data_arg.add_argument('--testdataset', type=str, default='ScannetVoxelization2cmtestDataset')
data_arg.add_argument('--temporal_dilation', type=int, default=30)
data_arg.add_argument('--temporal_numseq', type=int, default=3)
data_arg.add_argument('--point_lim', type=int, default=-1)
data_arg.add_argument('--pre_point_lim', type=int, default=-1)
data_arg.add_argument('--batch_size', type=int, default=1)
data_arg.add_argument('--test_batch_size', type=int, default=1)
data_arg.add_argument('--cache_data', type=str2bool, default=False)
data_arg.add_argument('--num_workers', type=int, default=0, help='num workers for train/test dataloader')
data_arg.add_argument('--num_val_workers', type=int, default=1, help='num workers for val dataloader')
data_arg.add_argument('--ignore_label', type=int, default=255)
data_arg.add_argument('--return_transformation', type=str2bool, default=False)
data_arg.add_argument('--ignore_duplicate_class', type=str2bool, default=False)
data_arg.add_argument('--partial_crop', type=float, default=0.)
data_arg.add_argument('--train_limit_numpoints', type=int, default=0)

# Point Cloud Dataset

data_arg.add_argument('--synthia_path', type=str, default='/home/chrischoy/datasets/Synthia/Synthia4D', help='Point Cloud dataset root dir')
# For temporal sequences
data_arg.add_argument('--temporal_rand_dilation', type=str2bool, default=False)
data_arg.add_argument('--temporal_rand_numseq', type=str2bool, default=False)

data_arg.add_argument('--scannet_path', type=str, default='/home/aidrive/luoly/datasets/Scannet/instance/20/train', help='Scannet online voxelization dataset root dir')

data_arg.add_argument('--scannet_test_path', type=str, default='/home/aidrive/luoly/datasets/Scannet/instance/full/train', help='Scannet online voxelization dataset root dir')

data_arg.add_argument('--stanford3d_path', type=str, default='/home/aidrive/luoly/datasets/S3DIS/Stanford_preprocessing/instance/full', help='Stanford precropped dataset root dir')

data_arg.add_argument('--stanford3d_test_path', type=str, default='/home/aidrive/luoly/datasets/S3DIS/Stanford_preprocessing/instance/full', help='Stanford precropped dataset root dir')

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--is_train', type=str2bool, default=True)
train_arg.add_argument('--stat_freq', type=int, default=40, help='print frequency')
train_arg.add_argument('--test_stat_freq', type=int, default=100, help='print frequency')
train_arg.add_argument('--test_stat_freq1', type=int, default=312, help='print frequency')
train_arg.add_argument('--save_freq', type=int, default=500, help='save frequency')
train_arg.add_argument('--val_freq', type=int, default=1000, help='validation frequency')
train_arg.add_argument('--empty_cache_freq', type=int, default=1, help='Clear pytorch cache frequency')
train_arg.add_argument('--train_phase', type=str, default='train', help='Dataset for training')
train_arg.add_argument('--val_phase', type=str, default='val', help='Dataset for validation')
train_arg.add_argument('--overwrite_weights', type=str2bool, default=True, help='Overwrite checkpoint during training')
train_arg.add_argument('--resume', default=None, type=str, help='path to latest checkpoint (default: none)')
train_arg.add_argument('--resume_optimizer', default=True, type=str2bool, help='Use checkpoint optimizer states when resume training')
train_arg.add_argument('--eval_upsample', type=str2bool, default=False)
train_arg.add_argument('--lenient_weight_loading', type=str2bool, default=False, help='Weights with the same size will be loaded')

# Distributed Training configurations
distributed_arg = add_argument_group('Distributed')
distributed_arg.add_argument('--distributed_world_size', type=int, default=1)
distributed_arg.add_argument('--distributed_rank', type=int, default=0)
distributed_arg.add_argument('--distributed_backend', type=str, default='nccl')
distributed_arg.add_argument('--distributed_init_method', type=str, default='')
distributed_arg.add_argument('--distributed_port', type=int, default=10010)
distributed_arg.add_argument('--device_id', type=int, default=0)
distributed_arg.add_argument('--distributed_no_spawn', type=str2bool, default=True)
distributed_arg.add_argument('--ddp_backend', type=str, default='c10d', choices=['c10d', 'no_c10d'])
distributed_arg.add_argument('--bucket_cap_mb', type=int, default=25)

# Data augmentation
data_aug_arg = add_argument_group('DataAugmentation')
data_aug_arg.add_argument('--use_feat_aug', type=str2bool, default=True, help='Simple feat augmentation')
data_aug_arg.add_argument('--data_aug_color_trans_ratio', type=float, default=0.10, help='Color translation range')
data_aug_arg.add_argument('--data_aug_color_jitter_std', type=float, default=0.05, help='STD of color jitter')
data_aug_arg.add_argument('--normalize_color', type=str2bool, default=True)
data_aug_arg.add_argument('--data_aug_scale_min', type=float, default=0.9)
data_aug_arg.add_argument('--data_aug_scale_max', type=float, default=1.1)
data_aug_arg.add_argument('--data_aug_hue_max', type=float, default=0.5, help='Hue translation range. [0, 1]')
data_aug_arg.add_argument('--data_aug_saturation_max', type=float, default=0.20, help='Saturation translation range, [0, 1]')

# Test
test_arg = add_argument_group('Test')
test_arg.add_argument('--visualize', type=str2bool, default=False)
test_arg.add_argument('--test_temporal_average', type=str2bool, default=False)
test_arg.add_argument('--visualize_path', type=str, default='outputs/visualize')
test_arg.add_argument('--save_prediction', type=str2bool, default=False)
test_arg.add_argument('--save_pred_dir', type=str, default='outputs/pred')
test_arg.add_argument('--test_phase', type=str, default='test', help='Dataset for test')
test_arg.add_argument('--evaluate_original_pointcloud', type=str2bool, default=False, help='Test on the original pointcloud space during network evaluation using voxel projection.')
test_arg.add_argument('--test_original_pointcloud', type=str2bool, default=False, help='Test on the original pointcloud space as given by the dataset using kd-tree.')

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--is_cuda', type=str2bool, default=True)
misc_arg.add_argument('--load_path', type=str, default='')
misc_arg.add_argument('--log_step', type=int, default=50)
misc_arg.add_argument('--log_level', type=str, default='INFO', choices=['INFO', 'DEBUG', 'WARN'])
misc_arg.add_argument('--num_gpu', type=str2bool, default=1)
misc_arg.add_argument('--seed', type=int, default=123)
misc_arg.add_argument('--run_name', type=str, default='instance')
misc_arg.add_argument('--unc_round', type=int, default=20)
misc_arg.add_argument('--unc_result_dir', type=str, default='/')

misc_arg.add_argument('--wandb', type=str2bool, default=False)
misc_arg.add_argument('--train_dataset', type=str, default='ScannetVoxelization2cmDataset')
misc_arg.add_argument('--val_dataset', type=str, default='ScannetVoxelization2cmtestDataset')
misc_arg.add_argument('--checkpoint_dir', type=str, default='checkpoints')
misc_arg.add_argument('--optim_step', type=int, default=1)
misc_arg.add_argument('--validate_step', type=int, default=100)
misc_arg.add_argument('--train_epoch', type=int, default=120000)
misc_arg.add_argument('--train_batch_size', type=int, default=4)
misc_arg.add_argument('--val_batch_size', type=int, default=1)
misc_arg.add_argument('--eval_result_dir', type=str, default='/')
misc_arg.add_argument('--unc_stat_path', type=str, default='/')
misc_arg.add_argument('--save_epoch', type=int, default=5)
misc_arg.add_argument('--ignore_index', type=int, default=255)
misc_arg.add_argument('--do_train', default=False, action='store_true')
misc_arg.add_argument('--do_validate', default=True, action='store_true')
misc_arg.add_argument('--do_unc_inference', action='store_true')
misc_arg.add_argument('--do_unc_demo', action='store_true')
misc_arg.add_argument('--unc_dataset', type=str, default="")
misc_arg.add_argument('--do_verbose_inference', action='store_true')
misc_arg.add_argument('--do_unc_render', action='store_true')
misc_arg.add_argument('--train_stuff', type=str2bool, default=False)
misc_arg.add_argument('--dual_set_cluster', type=str2bool, default=False)
misc_arg.add_argument('--evaluate_benchmark', type=str2bool, default=False)


def get_config():
    config = parser.parse_args()
    return config  # Training settings
