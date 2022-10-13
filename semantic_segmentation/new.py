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

from IPython import embed
import MinkowskiEngine as ME

import wandb

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
    num_labels = config.num_classes
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
        # wandb.init(project="3dscan", entity="one-one")
        wandb.init(project='spec-unc', entity='air-sun')
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
    elif config.do_validate:
        val_dataset_cls = load_dataset(config.val_dataset)
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
        val_loss, val_miou, iou_per_class = inference(model, val_dataloader, config, logger, rank=rank, world_size=world_size, evaluate=True)
        logger.info(f"VAL: loss (avg): {val_loss.item():.4f}, iou (avg): {val_miou.item():.4f}")
        for idx, i in enumerate(iou_per_class):
            logger.info(f"VAL: iou (cls#{idx}): {i.item():.4f}")
    elif config.do_unc_demo:
        val_dataset_cls = load_dataset(config.unc_dataset)
        unc_dataloader = initialize_data_loader(
            val_dataset_cls,
            config,
            num_workers=config.num_workers,
            phase='train',
            augment_data=False,
            shuffle=False,
            repeat=False,
            batch_size=config.test_batch_size,
            limit_numpoints=False,
        )
        unc_demo(model, unc_dataloader, config, logger)
        # unc_inference(model, unc_dataloader, config, logger)
    elif config.do_unc_inference:
        val_dataset_cls = load_dataset(config.unc_dataset)
        unc_dataloader = initialize_data_loader(
            val_dataset_cls,
            config,
            num_workers=config.num_workers,
            phase='train',
            augment_data=False,
            shuffle=False,
            repeat=False,
            batch_size=config.test_batch_size,
            limit_numpoints=False,
        )
        unc_inference(model, unc_dataloader, config, logger)
    elif config.do_verbose_inference:
        val_dataset_cls = load_dataset(config.unc_dataset)
        unc_dataloader = initialize_data_loader(
            val_dataset_cls,
            config,
            num_workers=config.num_workers,
            phase='train',
            augment_data=False,
            shuffle=False,
            repeat=False,
            batch_size=config.test_batch_size,
            limit_numpoints=False,
        )
        verbose_inference(model, unc_dataloader, config, logger)
    elif config.do_unc_render:
        val_dataset_cls = load_dataset(config.unc_dataset)
        unc_dataloader = initialize_data_loader(
            val_dataset_cls,
            config,
            num_workers=config.num_workers,
            phase='train',
            augment_data=False,
            shuffle=False,
            repeat=False,
            batch_size=config.test_batch_size,
            limit_numpoints=False,
        )
        unc_render(unc_dataloader, config, logger, option=2)
    if world_size > 1:
        dist.destroy_process_group()


def unc_demo(model, dataloader, config, logger, rank=0, world_size=1):
    """
    Run multiple inference and obtain uncertainty for each point
    Then save the uncertainty result
    """
    from plyfile import PlyData
    device = rank

    # Set model to eval mode except for the last dropout layer
    model.eval()
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            logger.warn(f"module {m.__class__.__name__} set to train")
            m.train()
    # TODO add this to config
    with open('splits/scannet/scannetv2_train.txt') as f:
        names = sorted([i.strip() for i in f.readlines()])
    with torch.no_grad():

        global_cnt = 0
        for step, batched_data in enumerate(dataloader):
            logger.info(f"{step}/{len(dataloader)} done")
            batched_coords, batched_feats, _ = batched_data

            # Normalize color
            batched_feats[:, :3] = batched_feats[:, :3] / 255. - 0.5

            # Feed forward
            batched_sparse_input = ME.SparseTensor(batched_feats.to(device), batched_coords.to(device))

            multiround_batched_scores = []
            for i in range(config.unc_round):
                logger.info(f"---> feed forward #{i} round")
                batched_sparse_output = model(batched_sparse_input)

                multiround_batched_scores.append(batched_sparse_output.F.cpu())

            multiround_batched_scores = torch.stack(multiround_batched_scores)
            # Batchsize * Pointcount * Classcount
            batched_scores = multiround_batched_scores.var(dim=0)
            for i in range(config.test_batch_size):
                logger.info(f"---> processing #{i} scene in the batch")
                selector = batched_coords[:, 0] == i
                single_scene_coords = batched_coords[selector][:, 1:]
                single_scene_scores = batched_scores[selector]
                save_prediction(single_scene_coords,
                                single_scene_scores,
                                f"{config.unc_result_dir}/{names[global_cnt].split('.')[0]}_prediction.ply",
                                mode='unc')
                global_cnt += 1
            #     single_scene_scores = batched_scores


def verbose_inference(model, dataloader, config, logger, rank=0, world_size=1):
    """
    For each scene, perform inference 10 (fixed) times to get:
    1. predicted value of 20 classes of this 10 times (illustrated with ply file)
    2. miou of 10 times
    3. unc table 
    For debugging, do this on one scene first

    required config: 
    - unc_round
    - test_batch_size
    - unc_result_dir

    WORKING !!!
    """
    device = rank

    # Switch model to evaluation mode, except for the dropout layer
    model.eval()
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            logger.warn(f"module {m.__class__.__name__} set to train")
            m.train()

    # Obtain scene names
    with open('splits/scannet/scannetv2_train.txt') as f:
        names = sorted([i.strip() for i in f.readlines()])

    with torch.no_grad():
        # FIXME Check validity of unc calculation!
        global_cnt = 0
        all_scores = []

        for step, batched_data in enumerate(dataloader):
            logger.info(f"{step}/{len(dataloader)} done")

            batched_coords, batched_feats, batched_target = batched_data

            # Normalize color
            batched_feats[:, :3] = batched_feats[:, :3] / 255. - 0.5

            # Feed forward
            batched_sparse_input = ME.SparseTensor(batched_feats.to(device), batched_coords.to(device))

            multiround_batched_scores = []

            for unc_round_id in range(config.unc_round):
                logger.info(f"---> feed forward #{unc_round_id} round")
                batched_sparse_output = model(batched_sparse_input)
                batched_output = batched_sparse_output.F.cpu().clone().detach()
                batched_labels = batched_output.argmax(dim=1)  # Select the class with highest score
                batched_scores = F.softmax(batched_output, dim=1)  # Use soft max along class dimension to obtain scores

                multiround_batched_scores.append(batched_scores)

                # for scene_id in range(config.test_batch_size):
                #     selector = batched_coords[:, 0] == scene_id
                #     single_scene_coords = batched_coords[selector][:, 1:]
                #     single_scene_labels = batched_labels[selector]
                #     single_scene_scores = batched_scores[selector]
                #     single_scene_target = batched_target[selector]

                # For each class, save the prediction image
                # save_prediction(single_scene_coords, single_scene_labels, f'{config.unc_result_dir}/test_scene{scene_id}_round#{unc_round_id}_prediction.ply', mode='prediction')
                # save_prediction(single_scene_coords, single_scene_target, f'{config.unc_result_dir}/test_round#{unc_round_id}_truth.ply', mode='prediction')
                # for class_id in range(20):
                #     class_name = CLASS_LABELS[class_id]
                #     input_feat = (single_scene_scores[:, class_id] - single_scene_scores[:, class_id].min()) / (single_scene_scores[:, class_id].max()-single_scene_scores[:, class_id].min())
                #     def in_fn(id, row, feat):
                #         return 127 + int(feat * 128), 255 - int(feat * 128), 127, 0
                #     save_prediction(single_scene_coords, input_feat, f'{config.unc_result_dir}/test_scene{scene_id}_cls#{class_id}_round#{unc_round_id}_{class_name}.ply', mode='class', in_fn=in_fn)
            tmp = torch.stack(multiround_batched_scores, dim=0)
            multi_scores = tmp.var(dim=0)
            multi_labels = tmp.sum(dim=0).argmax(dim=1)
            # embed()

            for scene_id in range(config.test_batch_size):
                selector = batched_coords[:, 0] == scene_id
                single_scene_coords = batched_coords[selector][:, 1:]
                single_scene_scores = multi_scores[selector]
                single_scene_labels = multi_labels[selector]

                def in_fn(id, row, feat):
                    return 127 + int(feat * 128), 255 - int(feat * 128), 127, 0

                # feats = single_scene_scores.min(dim=1)[0]
                feats = single_scene_scores.gather(dim=1, index=single_scene_labels.unflatten(0, (single_scene_labels.shape[0], 1)))

                # feats /= feats.max()
                coef = 4
                feats = (1 - torch.exp(-coef * feats / feats.max()))
                save_prediction(single_scene_coords, feats, f'{config.unc_result_dir}/test_scene{scene_id}_total_unc.ply', mode='else', in_fn=in_fn)
                for class_id in range(20):
                    class_name = CLASS_LABELS[class_id]
                    single_scene_scores[:, class_id] /= single_scene_scores[:, class_id].max()
                    save_prediction(single_scene_coords,
                                    single_scene_scores[:, class_id],
                                    f'{config.unc_result_dir}/test_scene{scene_id}_cls#{class_id}_{class_name}_unc.ply',
                                    mode='else',
                                    in_fn=in_fn)

            return


def unc_inference(model, dataloader, config, logger, rank=0, world_size=1):
    """
    DONE
    Run multiple inference and obtain uncertainty results for each point
    And for each scene:
    - save the statistics of the uncertainty in obj file (torch save)
    - save the predicted labelds in obj file (torch save)

    required config:
    - unc_round
    - unc_result_dir
    - test_batch_size

    call unc_render to obtain ply files
    """

    # Inference should be done on one GPU
    if world_size > 1 and rank > 0:
        return
    device = rank

    # Set model to eval mode except for the last dropout layer
    model.eval()
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            logger.warn(f"module {m.__class__.__name__} set to train")
            m.train()

    # Read scene names
    # Scenes from training datasets
    with open('splits/scannet/scannetv2_train.txt') as f:
        names = sorted([i.strip() for i in f.readlines()])

    with torch.no_grad():
        scene_cnt = 0
        for step, batched_data in enumerate(dataloader):
            logger.info(f"{step}/{len(dataloader)} done")

            # Load batched data
            batched_coords, batched_feats, _ = batched_data
            batched_feats[:, :3] = batched_feats[:, :3] / 255. - 0.5

            multiround_batched_scores = []

            # Feed forward for multiple rounds
            batched_sparse_input = ME.SparseTensor(batched_feats.to(device), batched_coords.to(device))
            for i in range(config.unc_round):
                logger.info(f"---> feed forward #{i} round")
                batched_sparse_output = model(batched_sparse_input)
                batched_output = batched_sparse_output.F.cpu()
                batched_scores = F.softmax(batched_output, dim=1)  # + IMPORTANT STEPS
                multiround_batched_scores.append(batched_scores)

            # batch_size * point_count * class_count
            multiround_batched_scores = torch.stack(multiround_batched_scores)

            batched_labels = multiround_batched_scores.sum(dim=0).argmax(dim=1)  # Label
            batched_scores = multiround_batched_scores.var(dim=0)  # Uncertainty

            # Save labels and uncertainty for each scene
            for scene_id in range(config.test_batch_size):
                logger.info(f"---> processing #{scene_id} scene in the batch")
                selector = batched_coords[:, 0] == scene_id
                single_scene_labels = batched_labels[selector]
                single_scene_scores = batched_scores[selector]

                torch.save(single_scene_labels.cpu(), f"{config.unc_result_dir}/{names[scene_cnt].split('.')[0]}_predicted.obj")
                torch.save(single_scene_scores.cpu(), f"{config.unc_result_dir}/{names[scene_cnt].split('.')[0]}_unc.obj")

                scene_cnt += 1


def render(path, coords, rgbs, labels):
    """
    Create a plyfile according to three the numpy arrays:
    coords: (N * 3) array of coordinates
    rgbs:   (N * 3) array of rgb values
    labels: (N) array of labels
    """
    from plyfile import PlyData, PlyElement
    assert len(coords) == len(rgbs)
    assert len(rgbs) == len(labels)
    point_num = len(coords)
    dtype = np.dtype([
        ('x', '<f4'),
        ('y', '<f4'),
        ('z', '<f4'),
        ('red', 'u1'),
        ('green', 'u1'),
        ('blue', 'u1'),
        ('label', 'u1'),
    ])
    vertex = np.ndarray((point_num,), dtype=dtype)

    vertex['x'] = coords[:, 0]
    vertex['y'] = coords[:, 1]
    vertex['z'] = coords[:, 2]
    vertex['red'] = rgbs[:, 0]
    vertex['green'] = rgbs[:, 1]
    vertex['blue'] = rgbs[:, 2]
    vertex['label'] = labels

    plyelement = PlyElement.describe(vertex, 'vertex')
    plydata = PlyData([plyelement], text=False)
    plydata.write(path)


def save_prediction(coords, feats, path, mode='unc', in_fn=None):
    """
    save prediction for **ONE** scene
    mode:
    - 'unc': treat feats as the uncertainty of each class (batchsize * num_cls)
    - 'prediction': treat feats as the predicted label (batchsize * 1)s
    """
    if mode == 'unc':
        # feats = feats.min(dim=1)[0]
        # feats = feats[:, 2]
        feats = feats.mean(dim=1)

        coef = 3  # exp smoothing
        feats = (1 - torch.exp(-coef * feats / feats.max())) * 128

        # feats = feats/feats.max() * 128

        def rgbl_fn(id, row, feat):
            return 127 + int(feat), 255 - int(feat), 127, 0
    elif mode == 'unc_select':
        feats, labels = feats
        feats = feats.gather(1, labels.unflatten(0, (labels.shape[0], 1)))
        coef = 3  # exp smoothing
        feats = (1 - torch.exp(-coef * feats / feats.max())) * 128

        def rgbl_fn(id, row, feat):
            return 127 + int(feat), 255 - int(feat), 127, 0
    elif mode == 'unc_reinforce':
        print('reinforce render')
        feats, labels, truths = feats
        feats = feats.gather(1, labels.unflatten(0, (labels.shape[0], 1)))
        coef = 3  # exp smoothing
        feats = (1 - torch.exp(-coef * feats / feats.max())) * 128

        def rgbl_fn(id, row, feat):
            if truths[id].item() == 255:
                return 127 + int(feat), 255 - int(feat), 64, 0
            else:
                return 0, 0, 255, 0

    elif mode == 'prediction':

        def rgbl_fn(id, row, feat):
            return (*[int(i) for i in COLOR_MAP[feat.item()]], feat)
    else:
        rgbl_fn = in_fn

    print(f'Saved to {path}')
    with open(path, 'w') as f:  # pylint: disable=invalid-name
        f.write('ply\n'
                'format ascii 1.0\n'
                f'element vertex {coords.shape[0]}\n'
                'property float x\n'
                'property float y\n'
                'property float z\n'
                'property uchar red\n'
                'property uchar green\n'
                'property uchar blue\n'
                'property uchar label\n'
                'end_header\n')
        for id, (row, feat) in enumerate(zip(coords, feats)):
            f.write(f'{row[0]} {row[1]} {row[2]} '
                    f'{rgbl_fn(id, row, feat)[0]} '
                    f'{rgbl_fn(id, row, feat)[1]} '
                    f'{rgbl_fn(id, row, feat)[2]} '
                    f'{rgbl_fn(id, row, feat)[3]}\n')


def fast_hist(predictions, labels, num_labels):
    selector = (labels >= 0) & (labels < num_labels)  # ndarray of bool type, selector
    stat = torch.bincount(num_labels * labels[selector].int() + predictions[selector], minlength=num_labels**2).reshape(num_labels, num_labels)
    return stat


def calc_iou(stat):
    with np.errstate(divide='ignore', invalid='ignore'):
        return stat.diagonal() / (stat.sum(dim=0) + stat.sum(dim=1) - stat.diagonal() + 1e-7)


def inference(model, dataloader, config, logger, rank=0, world_size=1, save=False, evaluate=True):
    """Evaluation on val / test dataset"""
    torch.cuda.empty_cache()
    device = f'cuda:{rank}'

    model.eval()

    with torch.no_grad():

        global_cnt = 0

        losses = []
        num_labels = dataloader.dataset.NUM_LABELS
        iu_hist = torch.zeros((num_labels, num_labels))
        # ious = []
        # ious_perclass = []
        # One epoch
        for step, (batched_coords, batched_feats, batched_targets) in enumerate(dataloader):

            # Normalize color
            batched_feats[:, :3] = batched_feats[:, :3] / 255. - 0.5

            # Feed forward
            batched_sparse_input = ME.SparseTensor(batched_feats.to(device), batched_coords.to(device))

            batched_sparse_output = model(batched_sparse_input)
            batched_outputs = batched_sparse_output.F.cpu()
            batched_prediction = batched_outputs.max(dim=1)[1].int()

            # if world_size == 1 or rank == 0:
            #     logger.info(f"---> inference step #{step} of {len(dataloader)} ")
            # Save predictions for each scene
            if save:
                for i in range(config.train_batch_size):
                    selector = batched_coords[:, 0] == i
                    single_scene_coords = batched_coords[selector][:, 1:]
                    single_scene_predictions = batched_prediction[selector]
                    save_prediction(single_scene_coords,
                                    single_scene_predictions,
                                    f"{config.eval_result_dir}/{global_cnt}_predicted_inference.ply",
                                    mode="prediction")
                    global_cnt += 1
            else:
                global_cnt += config.train_batch_size

            # Evaluate prediction
            if evaluate:
                # batched_targets = batched_targets.to(device)
                criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_index)
                loss = criterion(batched_outputs, batched_targets.long())
                iu_hist += fast_hist(batched_prediction.flatten(), batched_targets.flatten(), num_labels)
                iou = calc_iou(iu_hist)
                # ious_perclass.append(iou)

                if world_size == 1 or rank == 0:
                    logger.info(f"---> inference #{step} of {len(dataloader)} loss: {loss:.4f} iou: {iou.mean():.4f}")
                losses.append(loss)
                # ious.append(iou.mean())

                # TODO calculate average precision
                probablities = F.softmax(batched_outputs, dim=1)  # pylint: disable=unused-variable
                # avg_precision =

    if evaluate:
        # ious_perclass = torch.stack(ious_perclass).mean(dim=0)
        # return torch.stack(losses).mean(), torch.stack(ious).mean(), ious_perclass
        return torch.stack(losses).mean(), iou.mean(), iou


# !!! The output of the model should be .cpu()


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
    if config.resume and os.path.isfile(f"checkpoints/{config.run_name}_latest.pth"):
        states = torch.load(f"checkpoints/{config.run_name}_latest.pth", map_location=f"cuda:{device}")
        optimizer.load_state_dict(states['optimizer'])
        scheduler.load_state_dict(states['scheduler'])
        start_epoch = states['epoch'] + 1
        global_step = states['global_step']
        optim_step = states['optim_step']
        val_step = states['val_step']
        best_miou = states['best_miou']
        best_val_loss = states['val_loss']
    else:
        start_epoch = 0
        global_step = 0
        optim_step = 0
        val_step = 0
        best_miou = 0
        best_val_loss = float('inf')

    # TODO add metric meters
    total_step = 0
    losses = []
    precisions = []
    for epoch in range(start_epoch, config.train_epoch):
        for step, (coords, feats, targets) in enumerate(dataloader):
            # FIXME !!! Avoid train and val data taking space simultaneously

            torch.cuda.empty_cache()
            model.train()

            # TODO set seed here for certainty

            coords[:, 1:] += (torch.rand(3) * 100).type_as(coords)

            feats[:, :3] = feats[:, :3] / 255. - 0.5
            sparse_input = ME.SparseTensor(feats, coords, device=device)
            sparse_output = model(sparse_input)

            targets = targets.long().to(device)
            loss = criterion(sparse_output.F, targets)

            prediction = sparse_output.F.max(dim=1)[1]
            precision = prediction.eq(targets).view(-1).float().sum(dim=0).mul(100. / prediction.shape[0])

            loss /= config.optim_step
            loss.backward()
            losses.append(loss)
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
                # loss /= config.optim_step
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
                        f"TRAIN at global step #{global_step} Epoch #{epoch+1} Step #{step+1} / {len(dataloader)}: loss:{loss:.4f}, precision: {precision:.4f}"
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
                    val_loss, val_miou, iou_per_class = inference(model,
                                                                  val_dataloader,
                                                                  config,
                                                                  logger,
                                                                  rank=rank,
                                                                  world_size=world_size,
                                                                  evaluate=True)

                    if world_size == 1 or rank == 0:
                        logger.info(f"VAL   at global step #{global_step}: loss (avg): {val_loss.item():.4f}, iou (avg): {val_miou.item():.4f}")
                        for idx, i in enumerate(iou_per_class):
                            logger.info(f"VAL   at global step #{global_step}: iou (cls#{idx}): {i.item():.4f}")

                        if config.wandb:
                            obj = {
                                'val_loss': val_loss.cpu().item(),
                                'val_miou_mean': val_miou.cpu().item(),
                            }
                            wandb.log({
                                'val_step': val_step,
                                'global_step': global_step,
                                **obj,
                            })
                    if val_miou.item() > best_miou:
                        best_miou = val_miou.item()
                        best_val_loss = val_loss.item()
                        logger.info(f"Better checkpoint saved")
                        if world_size == 1 or rank == 0:
                            checkpoint(
                                model,
                                optimizer,
                                scheduler,
                                config,
                                prefix='best',
                                step=step,
                                best_miou=best_miou,
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


def stat(count=200):
    stats = []
    for count in (200, 100, 50, 20):
        try:
            stat = torch.load(f'results/{count}/stat_unc.obj')
            stats.append(stat)
        except:
            results_dir = f'results/{count}'
            with open('splits/scannet/scannetv2_train.txt') as f:
                names = sorted([i.strip() for i in f.readlines()])
            from tqdm import tqdm
            bins_num = 200
            stat = torch.zeros((20, bins_num)).cuda()
            for name in tqdm(names):
                # FIXME add mapping
                uncname = f"{results_dir}/{name.split('.')[0]}_unc.obj"
                labelname = f"{results_dir}/{name.split('.')[0]}_predicted.obj"
                unc = torch.load(uncname).cuda()
                labels = torch.load(labelname).cuda()
                for cls in range(20):
                    mask = labels == cls
                    stat[cls, :] += torch.histc(unc[:, cls][mask], bins_num, 0, 0.04)
            stat = stat.cpu()
            stats.append(stat)
            torch.save(stat, f'results/{count}/stat_unc.obj')
    save_hist(stats, f'./total_stat.png')
    return stats


def save_hist(ts, path: str):
    """
    TODO
    """
    import matplotlib.pyplot as plt
    ylims = [1e7, 1e7, 1e7, 1e6, 1e6, 1e6, 1e6, 8e5, 8e5, 5e5, 2e4, 2e4, 15e4, 6e5, 5e4, 1e5, 12e4, 5e4, 1e5, 1e6]
    plt.figure(figsize=(20, 4), dpi=1200)
    for idx, t in enumerate(ts):
        t = t.numpy()
        for cls in range(20):
            plt.subplot(4, 20, idx * 20 + cls + 1)
            plt.ylim(0, ylims[cls])
            plt.xticks([])
            plt.yticks([])
            plt.bar(range(len(t[cls, :])), t[cls, :])
    plt.tight_layout()
    plt.savefig(path, dpi=1200)


def extract_label():
    '''DEPRECATED'''
    results_dir = '/home/cloudroot/3DScanSeg/results/200'
    with open('splits/scannet/scannetv2_train.txt') as f:
        names = sorted([i.strip() for i in f.readlines()])
    from plyfile import PlyData
    from tqdm import tqdm
    for name in tqdm(names):
        plyname = f"{results_dir}/{name.split('.')[0]}_prediction.ply"
        plydata = PlyData.read(plyname)
        labels = torch.tensor(plydata['vertex']['label'])
        torch.save(labels, f"{results_dir}/{name.split('.')[0]}_predicted_labels.obj")


def unc_render(dataloader, config, logger, rank=0, world_size=1, option=1):
    if rank > 0:
        return
    from tqdm import tqdm
    from plyfile import PlyData
    global_cnt = 0
    with open('splits/scannet/scannetv2_train.txt') as f:
        names = sorted([i.strip() for i in f.readlines()])

    if option == 1:
        # Option 1: use dataloader to obtain coordinates
        for batched_coords, _, _ in dataloader:
            for scene_id in range(config.test_batch_size):
                selector = batched_coords[:, 0] == scene_id
                single_scene_coords = batched_coords[selector][:, 1:]
                name = names[global_cnt]
                logger.info(f'---> Processing {name}')

                uncs = torch.load(f'{config.unc_result_dir}/{name.split(".")[0]}_unc.obj')
                labels = torch.load(f'{config.unc_result_dir}/{name.split(".")[0]}_predicted.obj')
                coef = 4
                uncs = uncs.gather(1, labels.unflatten(0, (labels.shape[0], 1)))
                uncs = (1 - torch.exp(-coef * uncs / uncs.max()))

                rgbs = np.array(list(COLOR_MAP.values()))[labels]
                render(
                    f'{config.unc_result_dir}/{name.split(".")[0]}_predicted.ply',
                    single_scene_coords,
                    rgbs,
                    labels,
                )

                rgbs = np.ndarray((len(uncs), 3), dtype='u1')
                rgbs[:, 0] = 127 + uncs.flatten() * 128
                rgbs[:, 1] = 255 - uncs.flatten() * 128
                rgbs[:, 2] = 127
                render(
                    f'{config.unc_result_dir}/{name.split(".")[0]}_unc.ply',
                    single_scene_coords,
                    rgbs,
                    labels,
                )
                global_cnt += 1
    elif option == 2:
        # Option 2: use original pointcloud and obj to obtain coordinates
        path = ''
        for name in tqdm(names, desc=path):
            # FIXME fix this path
            path = f'/home/aidrive/tb5zhh/data/full_mesh/train/{name}'
            original = PlyData.read(path)
            uncs = torch.load(f'{config.unc_result_dir}/{name.split(".")[0]}_unc.obj')
            labels = torch.load(f'{config.unc_result_dir}/{name.split(".")[0]}_predicted.obj')
            coef = 4
            uncs = uncs.gather(1, labels.unflatten(0, (labels.shape[0], 1)))
            uncs = (1 - torch.exp(-coef * uncs / uncs.max()))

            inverse_mappings = torch.load(f'results/mappings/{name.split(".")[0]}_mapping.obj')['inverse']
            uncs = uncs[inverse_mappings]
            labels = labels[inverse_mappings]

            # prediction
            # original['vertex']['red'] = np.array(list(COLOR_MAP.values()))[labels][:, 0]
            # original['vertex']['green'] = np.array(list(COLOR_MAP.values()))[labels][:, 1]
            # original['vertex']['blue'] = np.array(list(COLOR_MAP.values()))[labels][:, 2]
            # original['vertex']['label'] = labels
            # original.write(f'{config.unc_result_dir}/{name.split(".")[0]}_predicted_raw.ply')

            original['vertex']['red'] = 24 * (1 - uncs.flatten()) + 255 * (uncs.flatten())
            original['vertex']['green'] = 96 * (1 - uncs.flatten()) + 128 * (uncs.flatten())
            original['vertex']['blue'] = 192 * (1 - uncs.flatten()) + 0 * (uncs.flatten())
            original['vertex']['label'] = labels
            # original.write(f'test_unc_raw.ply')
            original.write(f'{config.unc_result_dir}/{name.split(".")[0]}_unc_raw.ply')


if __name__ == '__main__':
    main()
    # stat(20)
    # extract_label()
