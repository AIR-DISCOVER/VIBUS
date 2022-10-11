import logging
import os
import time
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from lib.solvers import initialize_optimizer, initialize_scheduler
from lib.utils import save_predictions

from lib.dataset import initialize_data_loader
from lib.datasets import load_dataset
from lib.datasets.scannet import COLOR_MAP, CLASS_LABELS
from lib.distributed_utils import all_gather_list
from models import load_model
from config import get_config

from lib.bfs.bfs import Clustering
from lib.datasets.evaluation.evaluate_semantic_instance import Evaluator as InstanceEvaluator
from lib.test import test
import MinkowskiEngine as ME

import wandb

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def checkpoint(model, optimizer, scheduler, config, prefix='', world_size=1, **kwarg):
    """
    Save checkpoint of current model, optimizer, scheduler
    Other basic information are stored in kwargs
    """
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    filename = f"{config.checkpoint_dir}/{config.run_name}{('_' +  prefix) if len(prefix) > 0 else ''}.pth"
    states = {
        'state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),  # * load a GPU checkpoint to CPU
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'config': vars(config),
        **kwarg
    }

    torch.save(states, filename)


def setup_logger(config):
    """
    Logger setup function
    This function should only be called by main process in DDP
    """
    logging.root.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s][%(name)s\t][%(levelname)s\t] %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
    log_level = logging.INFO

    cli_handler = logging.StreamHandler()
    cli_handler.setFormatter(formatter)
    cli_handler.setLevel(log_level)

    if config.log_dir is not None:
        os.makedirs(config.log_dir, exist_ok=True)
        now = int(round(time.time() * 1000))
        timestr = time.strftime('%Y-%m-%d_%H:%M', time.localtime(now / 1000))
        filename = os.path.join(config.log_dir, f"{config.run_name}-{timestr}.log")

        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)

    logging.root.addHandler(cli_handler)
    logging.root.addHandler(file_handler)


def main():
    """
    Program entry
    Branch based on number of available GPUs
    """
    device_count = torch.cuda.device_count()
    if device_count > 1:
        port = random.randint(10000, 20000)
        init_method = f'tcp://localhost:{port}'
        mp.spawn(
            fn=main_worker,
            args=(device_count, init_method),
            nprocs=device_count,
        )
    else:
        main_worker()


def distributed_init(init_method, rank, world_size):
    """
    torch distributed iniitialized
    create a multiprocess group and initialize nccl communicator
    """
    if not dist.is_initialized():
        dist.init_process_group(
            backend='nccl',
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )
        if torch.cuda.is_available():
            dist.all_reduce(torch.zeros(1).cuda(rank))
        else:
            dist.all_reduce(torch.zeros(1))
        return dist.get_rank()
    logging.getLogger().warn("Distributed already initialized!")


def main_worker(rank=0, world_size=1, init_method=None):
    """
    Top pipeline
    """

    # + Device and distributed setup
    if not torch.cuda.is_available():
        raise Exception("No GPU Found")
    device = rank
    if world_size > 1:
        distributed_init(init_method, rank, world_size)

    config = get_config()
    setup_logger(config)
    logger = logging.getLogger(__name__)
    if rank == 0:
        logger.info(f'Run with {world_size} cpu')

    torch.cuda.set_device(device)

    # Set seed
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    if rank == 0:
        logger.info("Running config")
        for key, value in vars(config).items():
            logger.info(f"---> {key:>30}: {value}")  # pylint: disable=W1203

    # Setup model
    num_in_channel = 3  # RGB
    # num_labels = val_dataloader.dataset.NUM_LABELS
    num_labels = config.num_classes # Depends on datasets
    model_class = load_model(config.model)
    model = model_class(num_in_channel, num_labels, config)

    # Load pretrained weights
    if config.weights:
        state = torch.load(config.weights, map_location=f'cuda:{device}')
        model.load_state_dict({k: v for k, v in state['state_dict'].items() if not k.startswith('projector.')})
        if rank == 0:
            logger.info(f"Weights loaded from {config.weights}")  # pylint: disable=W1203
    if config.resume:
        try:
            state = torch.load(f"checkpoints/{config.run_name}_latest.pth", map_location=f'cuda:{device}')
            model.load_state_dict(state['state_dict'])
            if rank == 0:
                logger.info(f"Checkpoint resumed from {config.resume}")  # pylint: disable=W1203
        except Exception as e:
            logger.info(e)
            logger.warn(f"checkpoint not found")

    model = model.to(device)
    if rank == 0:
        logger.info("Model setup done")
        logger.info(f"\n{model}")  # pylint: disable=W1203

    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(
            module=model,
            device_ids=[device],
            output_device=[device],
            broadcast_buffers=False,
            # bucket_cap_mb=
        )
    if config.wandb and rank == 0:
        #wandb.login('6653cea7375b6dd706fe1386010d63e776edc4d4')
        wandb.init(project="spec-unc", entity="air-sun")
        wandb.run.name = f"{config.run_name}-{wandb.run.id}"
        wandb.run.save()
        wandb.config.update(config)
        # wandb.watch(model)

    # Action switch
    if config.do_train:
        # Set up test dataloader
        train_dataset_cls = load_dataset(config.train_dataset)
        val_dataset_cls = load_dataset(config.val_dataset)

        # hint: use phase to select different split of data
        train_dataloader = initialize_data_loader(
            train_dataset_cls,
            config,
            num_workers=config.num_workers,
            phase='train',
            augment_data=True,
            shuffle=True,
            repeat=True,
            batch_size=config.train_batch_size,
            limit_numpoints=False,
        )

        val_dataloader = initialize_data_loader(
            val_dataset_cls,
            config,
            num_workers=config.num_workers,
            phase='val',
            augment_data=False,
            shuffle=False,
            repeat=False,
            batch_size=config.val_batch_size,
            limit_numpoints=False,
        )
        if rank == 0:
            logger.info("Dataloader setup done")
        train(model, train_dataloader, val_dataloader, config, logger, rank=rank, world_size=world_size)
    if world_size > 1:
        dist.destroy_process_group()


def train(model, dataloader, val_dataloader, config, logger, rank=0, world_size=1):
    """TODO"""

    logger.info(f'My rank : {rank} / {world_size}')
    device = rank

    optimizer = initialize_optimizer(model.parameters(), config)
    scheduler = initialize_scheduler(optimizer, config)
    criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_label)

    if rank == 0:
        logger.info("Start training")

    # Load state dictionaries of optimizer and scheduler from checkpoint
    if config.resume and os.path.isfile(f"checkpoints/{config.run_name}_{global_step}_latest.pth"):
        states = torch.load(f"checkpoints/{config.run_name}_latest.pth", map_location=f"cuda:{device}")
        optimizer.load_state_dict(states['optimizer'])
        scheduler.load_state_dict(states['scheduler'])
        start_epoch = states['epoch'] + 1
        global_step = states['global_step']
        optim_step = states['optim_step']
        val_step = states['val_step']
        best_miou = states['best_miou']
        best_val_mAP = states['best_val_mAP']
        best_val_loss = states['val_loss']
    else:
        start_epoch = 0
        global_step = 0
        optim_step = 0
        val_step = 0
        best_miou = 0
        best_val_mAP = 0
        best_val_loss = float('inf')

    # TODO add metric meters
    total_step = 0
    losses = []
    precisions = []
    for epoch in range(start_epoch, config.train_epoch):
        for step, (coords, feats, targets, instances) in enumerate(dataloader):
            # FIXME !!! Avoid train and val data taking space simultaneously

            torch.cuda.empty_cache()
            model.train()

            # TODO set seed here for certainty

            coords[:, 1:] += (torch.rand(3) * 100).type_as(coords)
            feats[:, :3] = feats[:, :3] / 255. - 0.5
            sparse_input = ME.SparseTensor(feats, coords.int(), device=device)
            pt_offsets, sparse_output, _ = model(sparse_input)

            targets = targets.long().to(device)
            semantic_loss = criterion(sparse_output.F, targets)
            total_loss = semantic_loss

            #-----------------offset loss----------------------
            ## pt_offsets: (N, 3), float, cuda
            ## coords: (N, 3), float32
            ## centers: (N, 3), float32 tensor 
            ## instance_ids: (N), long
            centers = np.concatenate([instance['center'] for instance in instances])
            instance_ids = np.concatenate([instance['ids'] for instance in instances])

            centers = torch.from_numpy(centers).cuda()
            instance_ids = torch.from_numpy(instance_ids).cuda().long()

            gt_offsets = centers - coords[:,1:].cuda()   # (N, 3)
            gt_offsets *= dataloader.dataset.VOXEL_SIZE
            # gt_offsets *= 0.02
            pt_diff = pt_offsets.F - gt_offsets   # (N, 3)
            pt_dist = torch.sum(torch.abs(pt_diff), dim=-1)   # (N)
            valid = (instance_ids != -1).float()
            offset_norm_loss = torch.sum(pt_dist * valid) / (torch.sum(valid) + 1e-6)

            gt_offsets_norm = torch.norm(gt_offsets, p=2, dim=1)   # (N), float
            gt_offsets_ = gt_offsets / (gt_offsets_norm.unsqueeze(-1) + 1e-8)
            pt_offsets_norm = torch.norm(pt_offsets.F, p=2, dim=1)
            pt_offsets_ = pt_offsets.F / (pt_offsets_norm.unsqueeze(-1) + 1e-8)
            direction_diff = - (gt_offsets_ * pt_offsets_).sum(-1)   # (N)
            offset_dir_loss = torch.sum(direction_diff * valid) / (torch.sum(valid) + 1e-6)
            total_loss += offset_norm_loss + offset_dir_loss

            prediction = sparse_output.F.max(dim=1)[1]
            precision = prediction.eq(targets).view(-1).float().sum(dim=0).mul(100. / prediction.shape[0])

            total_loss /= config.optim_step
            total_loss.backward()
            losses.append(total_loss)
            precisions.append(precision)

            total_step += 1
            # periodic optimization
            if total_step % config.optim_step == 0:
                global_step += 1
                optim_step += 1
                loss = torch.tensor(losses).sum(dim=0).to(f"cuda:{device}")
                precision = torch.tensor(precisions).mean(dim=0).to(f"cuda:{device}")
                losses.clear()
                precisions.clear()
                optimizer.step()

                optimizer.zero_grad()
                scheduler.step()

                # Synchronize
                # obj = {'loss': loss.item(), 'precision': precision}
                if world_size > 1:
                    loss_list = [torch.zeros_like(loss) for _ in range(world_size)]
                    prec_list = [torch.zeros_like(precision) for _ in range(world_size)]
                    dist.all_gather(loss_list, loss)
                    dist.all_gather(prec_list, precision)
                    loss = torch.stack(loss_list).mean(dim=0)
                    precision = torch.stack(prec_list).mean(dim=0)
                    # obj = {k: np.mean([item[k] for item in obj]) for k in obj[0]}

                if world_size == 1 or rank == 0:

                    def get_lr(optimizer):
                        for param_group in optimizer.param_groups:
                            return param_group['lr']

                    logger.info(
                        f"TRAIN at global step #{global_step} Epoch #{epoch+1} Step #{step+1} / {len(dataloader)}: loss:{total_loss:.4f}, semantic:{semantic_loss:.4f}, instance_loss1:{offset_norm_loss:.4f}, instance_loss2:{offset_dir_loss:.4f}, precision: {precision:.4f}"
                    )
                    if config.wandb:
                        obj = {
                            'loss': loss.cpu().item(),
                            'precision': precision.cpu().item(),
                            'learning rate': get_lr(optimizer),
                        }
                        wandb.log({
                            'optim_step': optim_step,
                            'global_step': global_step,
                            **obj,
                        })

                del coords
                del feats
                del targets
                del sparse_input
                del sparse_output
                del loss
                del prediction
                del precision
                torch.cuda.empty_cache()

                # periodic evaluation
                # This step take extra GPU memory so clearup in advance is needed
                if optim_step % config.validate_step == 0 and rank == 0:
                    val_step += 1
                    val_loss, val_score, _, val_miou, iou_per_class, val_mAP = test(model, val_dataloader, config)

                    if world_size == 1 or rank == 0:
                        logger.info(f"VAL   at global step #{global_step}: loss (avg): {val_loss:.4f}, iou (avg): {val_miou.item():.4f}, mAP: {val_mAP.item():.4f}")
                        for idx, i in enumerate(iou_per_class):
                            logger.info(f"VAL   at global step #{global_step}: iou (cls#{idx}): {i.item():.4f}")

                        if config.wandb:
                            obj = {
                                'val_loss': val_loss,
                                'val_miou_mean': val_miou,
                                'val_score': val_score,
                                'val_mAP': val_mAP,
                            }
                            wandb.log({
                                'val_step': val_step,
                                'global_step': global_step,
                                **obj,
                            })
                    if val_miou.item() > best_miou:
                        best_miou = val_miou.item()
                        best_val_loss = val_loss
                        logger.info("Current best mIoU: {:.3f}".format(best_miou))
                        logger.info(f"Better checkpoint saved")
                        if world_size == 1 or rank == 0:
                            checkpoint(
                                model,
                                optimizer,
                                scheduler,
                                config,
                                prefix='best_miou',
                                step=step,
                                best_miou=best_miou,
                                best_val_mAP=best_val_mAP,
                                val_loss=val_loss,
                                epoch=epoch,
                                world_size=world_size,
                            )
                    if val_mAP.item() > best_val_mAP:
                        best_val_mAP = val_mAP.item()
                        best_val_loss = val_loss
                        logger.info("Current best mAP: {:.3f}".format(best_val_mAP))
                        logger.info(f"Better checkpoint saved")
                        if world_size == 1 or rank == 0:
                            checkpoint(
                                model,
                                optimizer,
                                scheduler,
                                config,
                                prefix='best_mAP',
                                step=step,
                                best_miou=best_miou,
                                best_val_mAP=best_val_mAP,
                                val_loss=val_loss,
                                epoch=epoch,
                                world_size=world_size,
                            )
            torch.cuda.empty_cache()

        # periodic checkpoint
        if (epoch + 1) % config.save_epoch == 0:
            if world_size == 1 or rank == 0:
                args = {
                    'best_miou': best_miou,
                    'best_mAP': best_val_mAP,
                    'val_loss': best_val_loss,
                    'epoch': epoch,
                    'global_step': global_step,
                    'optim_step': optim_step,
                    'val_step': val_step,
                }
                checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    config,
                    prefix='latest',
                    world_size=world_size,
                    **args,
                )
                logger.info(f"Checkpoint at epoch #{epoch} saved")
    # Note: these steps should be outside the func
    # TODO Calculate uncertainty, and store the results
    # TODO Obtain augmented results





if __name__ == '__main__':
    main()
