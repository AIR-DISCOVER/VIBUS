import os
import sys
import open3d as o3d
import numpy as np
import logging
import torch
import torch.multiprocessing as mp
try:
    mp.set_start_method('forkserver')  # Reuse process created
except RuntimeError:
    pass
import torch.distributed as dist
from config import get_config
from lib.test import test
from lib.mptrain import train
from lib.utils import load_state_with_same_shape, get_torch_device, count_parameters
from lib.dataset import initialize_data_loader
from lib.datasets import load_dataset
from models import load_model, load_wrapper
import MinkowskiEngine as ME

os.environ['CUDA_VISIBLE_DEVICES'] = '4, 7'

use_cuda = torch.cuda.is_available()
data_files = '/data/hdd01/luoly/Minkowski/scan_processed/train/'
voxel_size = 0.05
batch_size = 16


def main():
    # loss and network
    num_devices = torch.cuda.device_count()
    print(
        "Testing ",
        num_devices,
        " GPUs. Total batch size: ",
        num_devices * batch_size,
    )

    world_size = num_devices
    mp.spawn(main_worker, nprocs=num_devices, args=(num_devices, world_size))
    

def main_worker(device, ngpus_per_node, config):
    config = get_config()
    device = get_torch_device(config.is_cuda)
    num_devices = torch.cuda.device_count()
    world_size = num_devices
    rank = 0 * ngpus_per_node + num_devices - 1
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://localhost:23456",
        world_size=world_size,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
        rank=rank,
    )
    DatasetClass = load_dataset(data_files)

    logging.info('===> Initializing dataloader')
    if config.is_train:
        train_data_loader = initialize_data_loader(
            DatasetClass,
            config,
            phase=config.train_phase,
            num_workers=config.num_workers,
            augment_data=True,
            shuffle=True,
            repeat=True,
            batch_size=config.batch_size,
            limit_numpoints=config.train_limit_numpoints)

        val_data_loader = initialize_data_loader(
            DatasetClass,
            config,
            num_workers=config.num_val_workers,
            phase=config.val_phase,
            augment_data=False,
            shuffle=True,
            repeat=False,
            batch_size=config.val_batch_size,
            limit_numpoints=False)
        if train_data_loader.dataset.NUM_IN_CHANNEL is not None:
            num_in_channel = train_data_loader.dataset.NUM_IN_CHANNEL
        else:
            num_in_channel = 3  # RGB color

        num_labels = train_data_loader.dataset.NUM_LABELS
    else:
        test_data_loader = initialize_data_loader(
            DatasetClass,
            config,
            num_workers=config.num_workers,
            phase=config.test_phase,
            augment_data=False,
            shuffle=False,
            repeat=False,
            batch_size=config.test_batch_size,
            limit_numpoints=False)
        if test_data_loader.dataset.NUM_IN_CHANNEL is not None:
            num_in_channel = test_data_loader.dataset.NUM_IN_CHANNEL
        else:
            num_in_channel = 3  # RGB color

        num_labels = test_data_loader.dataset.NUM_LABELS

    logging.info('===> Building model')
    NetClass = load_model(config.model)
    if config.wrapper_type == 'None':
        model = NetClass(num_in_channel, num_labels, config)
        logging.info('===> Number of trainable parameters: {}: {}'.format(NetClass.__name__,
                                                                          count_parameters(model)))
    else:
        wrapper = load_wrapper(config.wrapper_type)
        model = wrapper(NetClass, num_in_channel, num_labels, config)
        logging.info('===> Number of trainable parameters: {}: {}'.format(
            wrapper.__name__ + NetClass.__name__, count_parameters(model)))

    logging.info(model)
    torch.cuda.set_device(device)
    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])
    model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)

    if config.weights == 'modelzoo':  # Load modelzoo weights if possible.
        logging.info('===> Loading modelzoo weights')
        model.preload_modelzoo()

    # Load weights if specified by the parameter.
    elif config.weights.lower() != 'none':
        logging.info('===> Loading weights: ' + config.weights)
        state = torch.load(config.weights)
        if config.weights_for_inner_model:
            model.model.load_state_dict(state['state_dict'])
        else:
            if config.lenient_weight_loading:
                matched_weights = load_state_with_same_shape(
                    model, state['state_dict'])
                model_dict = model.state_dict()
                model_dict.update(matched_weights)
                model.load_state_dict(model_dict)
            else:
                model.load_state_dict(state['state_dict'])

    if config.is_train:
        train(model, train_data_loader, val_data_loader, config)
    else:
        test(model, test_data_loader, config)


if __name__ == "__main__":
    main()
