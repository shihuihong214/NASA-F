# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Modified from AttentiveNAS (https://github.com/facebookresearch/AttentiveNAS)

import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import sys
import operator
from datetime import date
from tensorboardX import SummaryWriter
import torch.nn as nn
#from torch.utils.tensorboard import SummaryWriter
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torch
from data.data_loader import build_data_loader

from utils.config import setup
import utils.saver as saver
from utils.progress import AverageMeter, ProgressMeter, accuracy
import utils.comm as comm
import utils.logging as logging
from evaluate import attentive_nas_eval as attentive_nas_eval
from solver import build_optimizer, build_lr_scheduler
import models
from copy import deepcopy
import numpy as np

import loss_ops as loss_ops 


parser = argparse.ArgumentParser(description='AlphaNet Training')
parser.add_argument('--config-file', default=None, type=str, 
                    help='training configuration')
parser.add_argument('--machine-rank', default=0, type=int, 
                    help='machine rank, distributed setting')
parser.add_argument('--num-machines', default=1, type=int, 
                    help='number of nodes, distributed setting')
parser.add_argument('--dist-url', default="tcp://127.0.0.1:10001", type=str, 
                    help='init method, distributed setting')
parser.add_argument('--gpu', type=str, default='0', 
                    help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument("--port", type=str, default="10001")
parser.add_argument("--conv_ws", type=bool, default=False)
# parser.add_argument('--models_save_dir', default="ckpt/", type=str, 
                    # help='saved path'
# args = parser.parse_args()

logger = logging.get_logger(__name__)


def build_args_and_env(run_args):
    cudnn.benchmark = True
    assert run_args.config_file and os.path.isfile(run_args.config_file), 'cannot locate config file'
    args = setup(run_args.config_file)
    args.config_file = run_args.config_file
    args.gpu = run_args.gpu
    args.port = run_args.port
    args.conv_ws = run_args.conv_ws

    #load config
    # assert args.distributed and args.multiprocessing_distributed, 'only support DDP training'
    # args.distributed = True

    args.machine_rank = run_args.machine_rank
    args.num_nodes = run_args.num_machines
    # args.dist_url = run_args.dist_url
    args.models_save_dir = os.path.join(args.models_save_dir, args.exp_name)

    if not os.path.exists(args.models_save_dir):
        os.makedirs(args.models_save_dir)

    #backup config file
    saver.copy_file(args.config_file, '{}/{}'.format(args.models_save_dir, os.path.basename(args.config_file)))

    args.checkpoint_save_path = os.path.join(
        args.models_save_dir, 'alphanet.pth.tar'
    )
    args.logging_save_path = os.path.join(
        args.models_save_dir, f'stdout.log'
    )
    return args


def main():
    run_args = parser.parse_args()
    args = build_args_and_env(run_args)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    #cudnn.deterministic = True
    #warnings.warn('You have chosen to seed training. '
    #                'This will turn on the CUDNN deterministic setting, '
    #                'which can slow down your training considerably! '
    #                'You may see unexpected behavior when restarting '
    #                'from checkpoints.')
    gpu_ids = args.gpu.split(',')
    # print("gpu_ids",gpu_ids)
    args.gpu = []
    for gpu_id in gpu_ids:
        id = int(gpu_id)
        # print("id",id)
        args.gpu.append(id)
    gpu = args.gpu
    print("gpu",gpu)
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.num_nodes
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # raise NotImplementedError
        # args.world_size = ngpus_per_node * args.world_size
        main_worker(args.gpu, ngpus_per_node, args)
    
    # assert args.world_size > 1, 'only support ddp training'


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu  # local rank, local machine cuda id
    # args.local_rank = args.gpu
    # args.batch_size = args.batch_size_per_gpu
    # args.batch_size_total = args.batch_size * args.world_size
    #rescale base lr
    # args.lr_scheduler.base_lr = args.lr_scheduler.base_lr * (max(1, args.batch_size_total // 256))

    # set random seed, make sure all random subgraph generated would be the same
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu:
        torch.cuda.manual_seed(args.seed)

    # global_rank = args.gpu + args.machine_rank * ngpus_per_node
    # dist.init_process_group(
    #     backend=args.dist_backend, 
    #     init_method=args.dist_url,
    #     world_size=args.world_size, 
    #     rank=global_rank
    # )
    # print('args.distributed',args.distributed)
    if args.distributed:
        os.environ['MASTER_PORT'] = args.port
        dist.init_process_group(backend="nccl")

    # Setup logging format.
    logging.setup_logging(args.logging_save_path, 'w')
    logger_curve = SummaryWriter(args.models_save_dir)

    # logger.info(f"Use GPU: {args.gpu}, machine rank {args.machine_rank}, num_nodes {args.num_nodes}, \
    #                 gpu per node {ngpus_per_node}, world size {args.world_size}")

    # synchronize is needed here to prevent a possible timeout after calling
    # init_process_group
    # See: https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
    # comm.synchronize()

    # args.rank = comm.get_rank() # global rank
    # args.local_rank = args.gpu
    # torch.cuda.set_device(args.gpu)

    # build model
    logger.info("=> creating model '{}'".format(args.arch))
    model = models.model_factory.create_model(args)
    
    # model.cuda(args.gpu)
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        # if config.gpu is not None:
        if len(args.gpu) > 1:
            model.cuda()
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            # config.batch_size = int(config.batch_size / ngpus_per_node)
            args.num_workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=args.gpu, find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)

    else:
        model = torch.nn.DataParallel(model).cuda()
        

    # use sync batchnorm
    if getattr(args, 'sync_bn', False):
        model.apply(
                lambda m: setattr(m, 'need_sync', True))

    # model = comm.get_parallel_model(model, args.gpu) #local rank

    # logger.info(model)

    criterion = loss_ops.CrossEntropyLossSmooth(args.label_smoothing).cuda(args.gpu)
    soft_criterion = loss_ops.AdaptiveLossSoft(args.alpha_min, args.alpha_max, args.iw_clip).cuda(args.gpu)

    if not getattr(args, 'inplace_distill', True):
        soft_criterion = None

    ## load dataset, train_sampler: distributed
    train_loader, val_loader, train_sampler =  build_data_loader(args)
    args.n_iters_per_epoch = len(train_loader)

    # logger.info( f'building optimizer and lr scheduler, \
    #         local rank {args.gpu}, global rank {args.rank}, world_size {args.world_size}')
    
    optimizer = build_optimizer(args, model)
    # TODO:
    lr_scheduler = build_lr_scheduler(args, optimizer)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 400], gamma=0.1)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(config.nepochs), eta_min=config.learning_rate_min)
 
    # optionally resume from a checkpoint
    best_acc_min = 0
    best_acc_max = 0
    best_acc_random = 0
    best_epoch = 0

    if args.resume:
        best_acc_min, best_acc_max = saver.load_checkpoints(args, model, optimizer, lr_scheduler, logger)

    # logger.info(args)
    for epoch in range(args.start_epoch, args.epochs):
        # if args.distributed:
        #     train_sampler.set_epoch(epoch)

        args.curr_epoch = epoch
        logger.info('Training lr {}'.format(lr_scheduler.get_lr()[0]))

        # train for one epoch
        # FIXME:
        train_epoch(epoch, model, train_loader, optimizer, criterion, args, logger, logger_curve,soft_criterion=soft_criterion, lr_scheduler=lr_scheduler, conv_ws=args.conv_ws)


        # if comm.is_master_process() or args.distributed:
            # validate supernet model
        best_acc_min, best_acc_max, best_acc_random, is_best, best_epoch_min, best_epoch_max = validate(
            train_loader, val_loader, model, criterion, args,epoch, logger, logger_curve, best_acc_min, best_acc_max, best_acc_random, best_epoch
        )

        # if comm.is_master_process():
            # save checkpoints
        saver.save_checkpoint(
            args.checkpoint_save_path, 
            model,
            optimizer,
            lr_scheduler, 
            args,
            epoch,
            best_acc_min,
            best_acc_max,
            best_epoch_min, best_epoch_max,
            is_best = is_best
        )


def train_epoch(
    epoch, 
    model, 
    train_loader, 
    optimizer, 
    criterion, 
    args, logger, logger_curve, 
    soft_criterion=None, 
    lr_scheduler=None,
    conv_ws = False
):

    model.train()
    end = time.time()
    # print('ok')
    num_updates = epoch * len(train_loader)
    # print('ok')
    # print(len(train_loader))
    # print(train_loader)
    for batch_idx, (images, target) in enumerate(train_loader):
        # print()
        # measure data loading time
        # # print('ok')
        # data_time.update(time.time() - end)
        # print('ok')
        start_time = time.time()
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        data_time = time.time() - start_time

        # total subnets to be sampled
        num_subnet_training = max(2, getattr(args, 'num_arch_training', 2))
        optimizer.zero_grad()
        # print('ok')

        # FIXME: mix training 
        # flag = [100,300]
        flag = [250,300,350,400,450]
        # print('epoch:',epoch)
        if epoch < flag[0]:
            model.module.settings_for_mixture_training(pretrain_conv=True, pretrain_add=False, mix=False, mix_training=True)
        elif epoch < flag[1]:
            model.module.settings_for_mixture_training(pretrain_conv=False, pretrain_add=False, mix=True, mix_training=True, prob=0.1)
        elif epoch < flag[2]:
            model.module.settings_for_mixture_training(pretrain_conv=False, pretrain_add=False, mix=True, mix_training=True, prob=0.2)
        elif epoch < flag[3]:
            model.module.settings_for_mixture_training(pretrain_conv=False, pretrain_add=False, mix=True, mix_training=True, prob=0.3)
        elif epoch < flag[4]:
            model.module.settings_for_mixture_training(pretrain_conv=False, pretrain_add=False, mix=True, mix_training=True, prob=0.4)
        else:
            model.module.settings_for_mixture_training(pretrain_conv=False, pretrain_add=False, mix=True, mix_training=True, prob=0.5)
        
        # model.module.settings_for_mixture_training(pretrain_conv=False, pretrain_add=False, mix=False, mix_training=False, prob=0.3)

        ### compute gradients using sandwich rule ###
        # step 1 sample the largest network, apply regularization to only the largest network
        drop_connect_only_last_two_stages = getattr(args, 'drop_connect_only_last_two_stages', True)
        # print('before')
        model.module.sample_max_subnet()
        # print('before:',model.module.config)
        model.module.set_dropout_rate(args.dropout, args.drop_connect, drop_connect_only_last_two_stages) 
        # print("after:",model.module.config)
        #dropout for supernet
        output = model(images, conv_ws)
        loss = criterion(output, target)
        
        if conv_ws:
            kl_loss = model.module.get_kl_loss()
            loss += kl_loss

        loss.backward()

        with torch.no_grad():
            soft_logits = output.clone().detach()

        #step 2. sample the smallest network and several random networks
        sandwich_rule = getattr(args, 'sandwich_rule', True)
        model.module.set_dropout_rate(0, 0, drop_connect_only_last_two_stages)  #reset dropout rate
        
        # ################ minxture pretraining ######################
        for arch_id in range(1, num_subnet_training):
            if arch_id == num_subnet_training-1 and sandwich_rule:
                model.module.sample_min_subnet()
            else:
                model.module.sample_active_subnet()

            # calcualting loss
            output = model(images, conv_ws)
            # kl_loss = model.module.get_kl_loss()

            if soft_criterion:
                if epoch < flag[0]:
                    loss = soft_criterion(output, soft_logits)
                # # FIXME: kl_loss
                else:
                    loss = criterion(output, target)
                # loss = criterion(output, target)
            else:
                assert not args.inplace_distill
                loss = criterion(output, target)
            
            # loss += kl_loss

            loss.backward()

        # clip gradients if specfied
        if getattr(args, 'grad_clip_value', None):
            torch.nn.utils.clip_grad_value_(model.parameters(), args.grad_clip_value)

        optimizer.step()
        total_time = time.time() - start_time

        #accuracy measured on the local batch
        # TODO:
        # acc1, acc5 = accuracy(output, target, topk=(1, 5))
        
        # if args.distributed:
        #     corr1, corr5, loss = acc1*args.batch_size, acc5*args.batch_size, loss.item()*args.batch_size #just in case the batch size is different on different nodes
        #     stats = torch.tensor([corr1, corr5, loss, args.batch_size], device=args.gpu)
        #     dist.barrier()  # synchronizes all processes
        #     dist.all_reduce(stats, op=torch.distributed.ReduceOp.SUM) 
        #     corr1, corr5, loss, batch_size = stats.tolist()
        #     acc1, acc5, loss = corr1/batch_size, corr5/batch_size, loss/batch_size
        #     losses.update(loss, batch_size)
        #     top1.update(acc1, batch_size)
        #     top5.update(acc5, batch_size)
        # else:
        #     losses.update(loss.item(), images.size(0))
        #     top1.update(acc1, images.size(0))
        #     top5.update(acc5, images.size(0))
        # FIXME:
        if batch_idx % 20 == 0:
            if not (args.multiprocessing_distributed or args.distributed) or (args.distributed and dist.get_rank() == 0):
                # logger.info("[Epoch %d/%d][Step %d/%d] Loss=%.3f KL_Loss=%.3f Time=%.3f Data Time=%.3f" % (epoch + 1, args.epochs, batch_idx + 1, len(train_loader), loss.item()-kl_loss.item(), kl_loss.item(), total_time, data_time))
                logger.info("[Epoch %d/%d][Step %d/%d] Loss=%.3f Time=%.3f Data Time=%.3f" % (epoch + 1, args.epochs, batch_idx + 1, len(train_loader), loss.item(), total_time, data_time))
                logger_curve.add_scalar('loss/train', loss, epoch*len(train_loader)+batch_idx)
                # break

        # measure elapsed time
        # batch_time.update(time.time() - end)
        # end = time.time()

        num_updates += 1
        # TODO:
        # if (batch_idx+1) % 5 == 0:
        if lr_scheduler is not None:
            lr_scheduler.step()

        # if batch_idx % args.print_freq == 0:
        #     progress.display(batch_idx, logger)


def validate(
    train_loader, 
    val_loader, 
    model, 
    criterion, 
    args, 
    epoch,
    logger, logger_curve,
    best_acc_min, best_acc_max, best_acc_random, best_epoch,
    distributed = True,
):
    subnets_to_be_evaluated = {
        'attentive_nas_min_net': {},
        'attentive_nas_random_net': {},
        'attentive_nas_max_net': {},
    }

    best_acc_min, best_acc_max, best_acc_random, is_best, best_epoch_min, best_epoch_max = attentive_nas_eval.validate(
        subnets_to_be_evaluated,
        train_loader,
        val_loader, 
        model, 
        criterion,
        args,
        epoch, best_acc_min, best_acc_max, best_acc_random, best_epoch,
        logger_curve,
        logger,
        bn_calibration = True,
    )

    return best_acc_min, best_acc_max, best_acc_random, is_best, best_epoch_min, best_epoch_max



if __name__ == '__main__':
    main()


