from os import write
import numpy as np
import logging
import os.path as osp
import math
import scipy.ndimage
import torch
from torch import nn
from torch.serialization import default_restore_location
from tensorboardX import SummaryWriter

from lib.test import test
from lib.utils import checkpoint, precision_at_one, \
    Timer, AverageMeter, get_prediction, get_torch_device
from lib.solvers import initialize_optimizer, initialize_scheduler
from lib.distributed_utils import all_gather_list, get_world_size, get_rank
#from MinkowskiEngine import SparseTensor
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiOps as me
from IPython import embed

import pointnet2._ext as p2

def _set_seed(config, step):
    # Set seed based on args.seed and the update number so that we get
    # reproducible results when resuming from checkpoints
    seed = config.seed + step
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def validate(model, val_data_loader, writer, curr_iter, config, transform_data_fn):
    v_loss, v_score, v_mAP, v_mIoU = test(
        model, val_data_loader, config, transform_data_fn)
    writer.add_scalar('validation/mIoU', v_mIoU, curr_iter)
    writer.add_scalar('validation/loss', v_loss, curr_iter)
    writer.add_scalar('validation/precision_at_1', v_score, curr_iter)

    return v_mIoU


def load_state(model, state):
    if get_world_size() > 1:
        _model = model.module
    else:
        _model = model
    _model.load_state_dict(state)


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def lossforward(x, y):
    lambd = 3.9e-3
    x_mean = torch.mean(x, dim=0)
    y_mean = torch.mean(y, dim=0)
    x_std = torch.std(x, dim=0)
    y_std = torch.std(y, dim=0)
    x = torch.div(torch.sub(x, x_mean), x_std)
    y = torch.div(torch.sub(y, y_mean), y_std)
    up = torch.mm(x.t(), y)
    down1 = x.pow(2).sum(0, keepdim=True).sqrt()
    down2 = y.pow(2).sum(0, keepdim=True).sqrt()
    down = torch.mm(down1.t(), down2)
    cov = up / down
    #covnp = cov.cpu().detach().numpy()
    #np.save('/home/aidrive1/workspace/luoly/dataset/Min_scan/bt_train/cov/cov_%02d.npy' % (i), covnp)
    ret = cov - torch.eye(cov.shape[0]).cuda()
    on_diag = torch.diagonal(ret).add_(-1).pow_(2).sum().mul(1/512)
    off_diag = off_diagonal(ret).pow_(2).sum().mul(1/512)
    loss = on_diag + lambd * off_diag
    return loss


def train(model, data_loader, val_data_loader, config, transform_data_fn=None):
    device = config.device_id
    distributed = get_world_size() > 1
    #device = get_torch_device(config.is_cuda)
    # Set up the train flag for batch normalization
    model.train()

    # Configuration
    if not distributed or get_rank() == 0:
        writer = SummaryWriter(log_dir=config.log_dir)
    data_timer, iter_timer = Timer(), Timer()
    fw_timer, bw_timer, ddp_timer = Timer(), Timer(), Timer()

    data_time_avg, iter_time_avg = AverageMeter(), AverageMeter()
    fw_time_avg, bw_time_avg, ddp_time_avg = AverageMeter(), AverageMeter(), AverageMeter()

    losses, scores = AverageMeter(), AverageMeter()

    optimizer = initialize_optimizer(model.parameters(), config)
    scheduler = initialize_scheduler(optimizer, config)
    criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_label)

    # writer = SummaryWriter(log_dir=config.log_dir)

    # Train the network
    logging.info('===> Start training on {} GPUs, batch-size={}'.format(
        get_world_size(), config.batch_size * get_world_size()
    ))
    best_val_miou, best_val_iter, curr_iter, epoch, is_training = 0, 0, 1, 1, True

    if config.resume:
        checkpoint_fn = config.resume + '/weights.pth'
        if osp.isfile(checkpoint_fn):
            logging.info("=> loading checkpoint '{}'".format(checkpoint_fn))
            state = torch.load(checkpoint_fn, map_location=lambda s,
                               l: default_restore_location(s, 'cpu'))
            curr_iter = state['iteration'] + 1
            epoch = state['epoch']
            model.load_state_dict(state['state_dict'])
            if config.resume_optimizer:
                scheduler = initialize_scheduler(
                    optimizer, config, last_step=curr_iter)
                optimizer.load_state_dict(state['optimizer'])
            if 'best_val' in state:
                best_val_miou = state['best_val']
                best_val_iter = state['best_val_iter']
            logging.info("=> loaded checkpoint '{}' (epoch {})".format(
                checkpoint_fn, state['epoch']))
        else:
            raise ValueError(
                "=> no checkpoint found at '{}'".format(checkpoint_fn))

    data_iter = data_loader.__iter__()
    while is_training:
        for iteration in range(len(data_loader) // config.iter_size):
            optimizer.zero_grad()
            data_time, batch_loss, batch_score = 0, 0, 0
            iter_timer.tic()

            # set random seed for every iteration for trackability
            _set_seed(config, curr_iter)

            for sub_iter in range(config.iter_size):
                # Get training data
                data_timer.tic()
                coords1, coords2, input1, target1, input2, target2 = data_iter.next()

                # For some networks, making the network invariant to even, odd coords is important. Random translation
                coords1[:, 1:] += (torch.rand(3) * 100).type_as(coords1)
                coords2[:, 1:] += (torch.rand(3) * 100).type_as(coords2)

                # Preprocess input
                color1 = input1[:, :3].int()
                color2 = input2[:, :3].int()
                if config.normalize_color:
                    input1[:, :3] = input1[:, :3] / 255. - 0.5
                    input2[:, :3] = input2[:, :3] / 255. - 0.5
                #sinput1 = ME.SparseTensor(input1.to(device), coords1.int().to(device))
                #sinput2 = ME.SparseTensor(input2.to(device), coords2.int().to(device))

                tfield1 = ME.TensorField(coordinates=coords1.int().to(device), features=input1.to(
                    device), quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)
                tfield2 = ME.TensorField(coordinates=coords2.int().to(device), features=input2.to(
                    device), quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)

                # print(len(tfield1))  # 227742
                # print(len(tfield2))
                sinput1 = tfield1.sparse()  # 161890 quantization results in fewer voxels
                sinput2 = tfield2.sparse()

                #sinput = ME.SparseTensor(input.to(device), coords.to(device))
                #sinput = ME.SparseTensor(input,coords).to(device)
                data_time += data_timer.toc(False)

                # Feed forward
                fw_timer.tic()

                # inputs = (sinput,) if config.wrapper_type == 'None' else (
                # sinput, coords, color)
                inputs1 = (sinput1,) if config.wrapper_type == 'None' else (
                    sinput1, coords1, color1)
                inputs2 = (sinput2,) if config.wrapper_type == 'None' else (
                    sinput2, coords2, color2)
                # model.initialize_coords(*init_args)
                #soutput = model(*inputs)
                soutput1 = model(*inputs1)
                soutput2 = model(*inputs2)

                # print(len(soutput1))  # 161890 Output with the same resolution
                # print(len(soutput2))
                ofield1 = soutput1.slice(tfield1)
                ofield2 = soutput2.slice(tfield2)
                #assert isinstance(ofield1, ME.TensorField)
                # len(ofield1) == len(coords1)  # recovers the original ordering and length
                #assert isinstance(ofield1.F, torch.Tensor)
                #assert isinstance(ofield2, ME.TensorField)
                # len(ofield2) == len(coords2)  # recovers the original ordering and length
                #assert isinstance(ofield2.F, torch.Tensor)

                # The output of the network is not sorted
                target1 = target1.long().to(device)
                target2 = target2.long().to(device)

                pindex = torch.randint(
                    0, (ofield1.F).shape[0], (1024,)).to(device)
                # logging.warn(ofield1.C.shape)
                pindex1 = p2.furthest_point_sampling(ofield1.C[:, 1:].reshape((1,ofield1.C.shape[0],3)).contiguous(), 1024).reshape(1024).long()
                pindex2 = p2.furthest_point_sampling(ofield2.C[:, 1:].reshape((1,ofield2.C.shape[0],3)).contiguous(), 1024).reshape(1024).long()

                # embed()
                # logging.warn(pindex1.shape)
                # print(pindex.shape)
                # print("=====")
                #list1 = soutput1.F[ps1]
                list1 = torch.index_select(ofield1.F, 0, pindex1)
                list2 = torch.index_select(ofield2.F, 0, pindex1)
                #list2 = torch.index_select(soutput2.F, 0, pshape)
                #loss = criterion(soutput.F, target.long())
                #loss = lossforward(soutput1.F,soutput2.F)
                loss = lossforward(list1, list2)

                # Compute and accumulate gradient
                loss /= config.iter_size

                pred = get_prediction(data_loader.dataset, soutput1.F, target1)
                score = precision_at_one(pred, target1)

                fw_timer.toc(False)
                bw_timer.tic()

                loss.backward()
                bw_timer.toc(False)

                logging_output = {
                    'loss': loss.item(), 'score': score / config.iter_size}

                ddp_timer.tic()
                if distributed:
                    logging_output = all_gather_list(logging_output)
                    logging_output = {w: np.mean([
                        a[w] for a in logging_output]
                    ) for w in logging_output[0]}

                batch_loss += logging_output['loss']
                batch_score += logging_output['score']
                ddp_timer.toc(False)

            # Update number of steps
            optimizer.step()
            scheduler.step()

            data_time_avg.update(data_time)
            iter_time_avg.update(iter_timer.toc(False))
            fw_time_avg.update(fw_timer.diff)
            bw_time_avg.update(bw_timer.diff)
            ddp_time_avg.update(ddp_timer.diff)

            losses.update(batch_loss, target1.size(0))
            scores.update(batch_score, target1.size(0))

            # for name, parms in model.named_parameters():
            #    print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data), ' -->grad_value:', torch.mean(parms.grad))

            if curr_iter >= config.max_iter:
                is_training = False
                break

            if curr_iter % config.stat_freq == 0 or curr_iter == 1:
                lrs = ', '.join(['{:.3e}'.format(x)
                                for x in scheduler.get_lr()])
                debug_str = "===> Epoch[{}]({}/{}): Loss {:.4f}\tLR: {}\t".format(
                    epoch, curr_iter,
                    len(data_loader) // config.iter_size, losses.avg, lrs)
                debug_str += "Score {:.3f}\tData time: {:.4f}, Forward time: {:.4f}, Backward time: {:.4f}, DDP time: {:.4f}, Total iter time: {:.4f}".format(
                    scores.avg, data_time_avg.avg, fw_time_avg.avg, bw_time_avg.avg, ddp_time_avg.avg, iter_time_avg.avg)
                logging.info(debug_str)
                # Reset timers
                data_time_avg.reset()
                iter_time_avg.reset()
                # Write logs
                if not distributed or get_rank() == 0:
                    writer.add_scalar('training/loss', losses.avg, curr_iter)
                    writer.add_scalar('training/precision_at_1',
                                      scores.avg, curr_iter)
                    writer.add_scalar('training/learning_rate',
                                      scheduler.get_lr()[0], curr_iter)
                losses.reset()
                scores.reset()

            # Save current status, save before val to prevent occational mem overflow
            if curr_iter % config.save_freq == 0:
                checkpoint(model, optimizer, epoch, curr_iter,
                           config, best_val_miou, best_val_iter)

            # Validation
            if curr_iter % config.val_freq == 0 and (not distributed or get_rank() == 0):
                val_miou = validate(model, val_data_loader,
                                    writer, curr_iter, config, transform_data_fn)
                if val_miou > best_val_miou:
                    best_val_miou = val_miou
                    best_val_iter = curr_iter
                    checkpoint(model, optimizer, epoch, curr_iter, config, best_val_miou, best_val_iter,
                               "best_val")
                logging.info("Current best mIoU: {:.3f} at iter {}".format(
                    best_val_miou, best_val_iter))

                # Recover back
                model.train()

            if curr_iter % config.empty_cache_freq == 0:
              # Clear cache
                torch.cuda.empty_cache()

            # End of iteration
            curr_iter += 1

        epoch += 1

    # Explicit memory cleanup
    if hasattr(data_iter, 'cleanup'):
        data_iter.cleanup()

    # Save the final model
    checkpoint(model, optimizer, epoch, curr_iter,
               config, best_val_miou, best_val_iter)
    if not distributed or get_rank() == 0:
        val_miou = validate(model, val_data_loader, writer,
                            curr_iter, config, transform_data_fn)
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            best_val_iter = curr_iter
            checkpoint(model, optimizer, epoch, curr_iter, config,
                    best_val_miou, best_val_iter, "best_val")
        writer.close()
    logging.info("Current best mIoU: {:.3f} at iter {}".format(
        best_val_miou, best_val_iter))
