# Change dataloader multiprocess start method to anything not fork
import open3d as o3d
import numpy as np
import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn')  # Reuse process created
except RuntimeError:
    pass
import torch.distributed as dist
import os
import sys
import json
import random
import logging
from easydict import EasyDict as edict

# Torch packages
import torch
from torch.serialization import default_restore_location
# Train deps
from config import get_config

from lib.test import test
from lib.trainsave import train
from lib import distributed_utils
from lib.utils import load_state_with_same_shape, get_torch_device, count_parameters
from lib.dataset import initialize_data_loader
from lib.datasets import load_dataset

from models import load_model, load_wrapper


os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

def setup_logging(config):
    ch = logging.StreamHandler(sys.stdout)
    if config.distributed_world_size > 1 and config.distributed_rank > 0:
        logging.getLogger().setLevel(logging.WARN)
    else:
        logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(
        format=os.uname()[1].split('.')[0] + ' %(asctime)s %(message)s',
        datefmt='%m/%d %H:%M:%S',
        handlers=[ch])


def main():
    config = get_config()
    print('main')
    num_devices = torch.cuda.device_count()
    print(
        "Testing ",
        num_devices,
        " GPUs. Total batch size: ",
        num_devices * config.batch_size,
    )
    if torch.cuda.device_count() > 1:
        #start_rank = config.distributed_rank
        #config.distributed_rank = None  # assign automatically
        #mp.spawn(
            #fn=main_worker,
            #args=(config, start_rank),
            #nprocs=num_devices,
        #)
        #assert config.distributed_world_size <= torch.cuda.device_count()
        port = random.randint(10000, 20000)
        config.distributed_init_method = 'tcp://localhost:{port}'.format(port=port)
        config.distributed_rank = None  # set based on device id

        mp.spawn(
            fn=distributed_main,
            args=(config, ),
            nprocs=config.distributed_world_size,
    )


def distributed_main(i, config, start_rank=0):
  config.device_id = i
  if config.distributed_rank is None:  # torch.multiprocessing.spawn
      config.distributed_rank = start_rank + i
  main_worker(config, init_distributed=True)

def main_worker(config, init_distributed=False):
    #config = get_config()
    if config.resume:
        json_config = json.load(open(config.resume + '/config.json', 'r'))
        json_config['resume'] = config.resume
        config = edict(json_config)

    if config.is_cuda and not torch.cuda.is_available():
        raise Exception("No GPU found")
    # device = get_torch_device(config.is_cuda)

        # setup initial seed
    torch.cuda.set_device(config.device_id)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    device = config.device_id
    distributed = config.distributed_world_size > 1

    if init_distributed:
        config.distributed_rank = distributed_utils.distributed_init(config)
    #if init_distributed:
        #dist.init_process_group(
            #backend="nccl",
            #init_method=config.distributed_init_method,
            #world_size=config.distributed_world_size,
            #rank=torch.distributed.get_rank(),
        #)  

    setup_logging(config)
    logging.info('===> Configurations')
    dconfig = vars(config)
    for k in dconfig:
        logging.info('    {}: {}'.format(k, dconfig[k]))

    DatasetClass = load_dataset(config.dataset)
    testDatasetClass = load_dataset(config.testdataset)
    if config.test_original_pointcloud:
        if not DatasetClass.IS_FULL_POINTCLOUD_EVAL:
            raise ValueError(
                'This dataset does not support full pointcloud evaluation.')

    if config.evaluate_original_pointcloud:
        if not config.return_transformation:
            raise ValueError(
                'Pointcloud evaluation requires config.return_transformation=true.')

    if (config.return_transformation ^ config.evaluate_original_pointcloud):
        raise ValueError('Rotation evaluation requires config.evaluate_original_pointcloud=true and '
                         'config.return_transformation=true.')

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
            testDatasetClass,
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
            testDatasetClass,
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
    # model = model.to(device)

    if config.weights == 'modelzoo':  # Load modelzoo weights if possible.
        logging.info('===> Loading modelzoo weights')
        model.preload_modelzoo()

    # Load weights if specified by the parameter.
    elif config.weights.lower() != 'none':
        logging.info('===> Loading weights: ' + config.weights)
        # state = torch.load(config.weights)
        state = torch.load(config.weights, map_location=lambda s,
                           l: default_restore_location(s, 'cpu'))

        if 'state_dict' in state.keys():
            state_key_name = 'state_dict'
        elif 'model_state' in state.keys():
            state_key_name = 'model_state'
        else:
            raise NotImplementedError

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
    # model = model.to(device)
    model = model.to(device)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[device], output_device=device,
            broadcast_buffers=False, bucket_cap_mb=config.bucket_cap_mb
        )

    if config.is_train:
        train(model, train_data_loader, val_data_loader, config)
    else:
        test(model, test_data_loader, config)


if __name__ == '__main__':
    main()
    # __spec__ = None
    # main()
    #os.environ['MKL_THREADING_LAYER'] = 'GNU'
