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
parser.add_argument("--quant", action='store_true', default=False)
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
    # print("gpu",gpu)
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
    
    # optimizer = build_optimizer(args, model)
    # lr_scheduler = build_lr_scheduler(args, optimizer)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(config.nepochs), eta_min=config.learning_rate_min)
 
    # optionally resume from a checkpoint
    best_acc = 0
    best_epoch = 0

    # print('args.resume:',args.resume)
    if args.resume:
        best_acc_min, best_acc_max = saver.load_checkpoints(args, model)
        
        cfg = [{'resolution': 32, 'width': [16, 16, 32, 40, 72, 120, 208, 216, 1984], 'depth': [1, 5, 3, 5, 3, 3, 1], 'kernel_size': [3, 5, 5, 5, 5, 5, 3], 'expand_ratio': [1, 6, 6, 6, 4, 6, 6], 'type': [['conv', 'conv'], ['conv', 'conv', 'conv', 'conv', 'conv'], ['conv', 'conv', 'conv'], ['conv', 'conv', 'add', 'add', 'conv'], ['add', 'add', 'add'], ['add', 'conv', 'add'], ['add']], 'Energy': 0.162, 'Latency': 2.056, 'FLOPs': 4.302, 'acc': 62.81}, {'resolution': 32, 'width': [16, 16, 32, 40, 64, 120, 208, 216, 1984], 'depth': [1, 5, 3, 5, 3, 3, 1], 'kernel_size': [3, 5, 5, 5, 3, 5, 3], 'expand_ratio': [1, 6, 6, 6, 4, 6, 6], 'type': [['conv', 'conv'], ['conv', 'conv', 'conv', 'conv', 'conv'], ['conv', 'conv', 'conv'], ['conv', 'conv', 'add', 'add', 'conv'], ['add', 'add', 'add'], ['add', 'add', 'add'], ['add']], 'Energy': 0.157, 'Latency': 1.939, 'FLOPs': 4.062, 'acc': 62.78}, {'resolution': 32, 'width': [16, 16, 32, 40, 64, 120, 208, 216, 1984], 'depth': [1, 5, 3, 5, 3, 3, 1], 'kernel_size': [3, 5, 5, 5, 3, 5, 3], 'expand_ratio': [1, 6, 6, 6, 4, 6, 6], 'type': [['conv', 'conv'], ['conv', 'conv', 'conv', 'conv', 'conv'], ['conv', 'conv', 'conv'], ['conv', 'conv', 'add', 'add', 'conv'], ['add', 'add', 'add'], ['add', 'conv', 'add'], ['add']], 'FLOPs': 4.203, 'acc': 62.77, 'Energy': 0.159, 'Latency': 1.984}, {'resolution': 32, 'width': [16, 16, 32, 40, 64, 120, 208, 216, 1984], 'depth': [1, 5, 3, 5, 3, 5, 1], 'kernel_size': [3, 5, 5, 5, 3, 5, 3], 'expand_ratio': [1, 6, 6, 5, 4, 6, 6], 'type': [['conv', 'add'], ['conv', 'conv', 'conv', 'conv', 'conv'], ['conv', 'conv', 'conv'], ['conv', 'conv', 'conv', 'add', 'conv'], ['add', 'add', 'add'], ['add', 'conv', 'add', 'add', 'conv'], ['add']], 'Energy': 0.185, 'Latency': 2.332, 'FLOPs': 4.635, 'acc': 62.73}, {'resolution': 32, 'width': [16, 16, 32, 40, 72, 120, 208, 224, 1792], 'depth': [2, 5, 3, 5, 4, 5, 1], 'kernel_size': [5, 5, 5, 5, 5, 3, 3], 'expand_ratio': [1, 6, 6, 5, 4, 6, 6], 'type': [['conv', 'add'], ['conv', 'conv', 'conv', 'conv', 'conv'], ['conv', 'conv', 'conv'], ['conv', 'conv', 'add', 'add', 'conv'], ['add', 'add', 'add', 'conv'], ['add', 'conv', 'add', 'add', 'conv'], ['add']], 'Energy': 0.199, 'Latency': 2.152, 'FLOPs': 4.943, 'acc': 62.71}, {'resolution': 32, 'width': [16, 16, 32, 40, 72, 120, 208, 224, 1792], 'depth': [2, 5, 3, 5, 3, 5, 1], 'kernel_size': [3, 5, 5, 5, 5, 3, 3], 'expand_ratio': [1, 6, 6, 6, 4, 6, 6], 'type': [['conv', 'conv'], ['conv', 'conv', 'conv', 'conv', 'conv'], ['conv', 'conv', 'conv'], ['conv', 'conv', 'add', 'add', 'conv'], ['add', 'add', 'add'], ['conv', 'conv', 'add', 'add', 'conv'], ['add']], 'Energy': 0.192, 'Latency': 2.366, 'FLOPs': 4.892, 'acc': 62.66}, {'resolution': 32, 'width': [16, 16, 32, 40, 72, 120, 208, 224, 1792], 'depth': [2, 5, 3, 5, 3, 3, 1], 'kernel_size': [5, 5, 5, 5, 3, 3, 3], 'expand_ratio': [1, 5, 6, 5, 4, 6, 6], 'type': [['conv', 'conv'], ['conv', 'conv', 'conv', 'conv', 'conv'], ['conv', 'conv', 'conv'], ['conv', 'conv', 'conv', 'add', 'conv'], ['add', 'add', 'add'], ['add', 'conv', 'add'], ['add']], 'FLOPs': 3.982, 'acc': 62.02, 'Energy': 0.153, 'Latency': 1.834}, {'resolution': 32, 'width': [16, 16, 32, 40, 64, 120, 208, 216, 1984], 'depth': [2, 5, 3, 5, 3, 3, 1], 'kernel_size': [5, 5, 5, 5, 3, 5, 3], 'expand_ratio': [1, 5, 6, 5, 4, 6, 6], 'type': [['conv', 'add'], ['conv', 'conv', 'conv', 'conv', 'conv'], ['conv', 'conv', 'conv'], ['conv', 'conv', 'conv', 'add', 'conv'], ['add', 'add', 'add'], ['add', 'conv', 'add'], ['add']], 'Energy': 0.15, 'Latency': 1.673, 'FLOPs': 3.882, 'acc': 61.95}, {'resolution': 32, 'width': [16, 16, 32, 40, 72, 112, 192, 224, 1984], 'depth': [2, 5, 3, 5, 3, 3, 1], 'kernel_size': [5, 5, 5, 3, 3, 5, 3], 'expand_ratio': [1, 5, 6, 6, 4, 6, 6], 'type': [['conv', 'add'], ['conv', 'conv', 'conv', 'conv', 'conv'], ['conv', 'conv', 'conv'], ['add', 'conv', 'conv', 'add', 'conv'], ['add', 'add', 'add'], ['add', 'conv', 'add'], ['add']], 'Energy': 0.144, 'Latency': 1.587, 'FLOPs': 3.828, 'acc': 61.77}, {'resolution': 32, 'width': [24, 16, 32, 40, 72, 120, 208, 224, 1984], 'depth': [2, 5, 3, 5, 3, 3, 1], 'kernel_size': [5, 3, 5, 5, 3, 5, 3], 'expand_ratio': [1, 5, 4, 5, 4, 6, 6], 'type': [['conv', 'conv'], ['conv', 'conv', 'conv', 'add', 'conv'], ['conv', 'conv', 'conv'], ['conv', 'conv', 'conv', 'add', 'conv'], ['conv', 'conv', 'add'], ['add', 'conv', 'conv'], ['add']], 'FLOPs': 3.673, 'acc': 55.89, 'Energy': 0.14, 'Latency': 1.548}, {'resolution': 32, 'width': [16, 24, 24, 32, 72, 120, 192, 224, 1792], 'depth': [2, 5, 3, 4, 6, 8, 2], 'kernel_size': [5, 5, 5, 5, 3, 3, 3], 'expand_ratio': [1, 6, 6, 5, 5, 6, 6], 'type': [['add', 'conv'], ['conv', 'add', 'add', 'conv', 'add'], ['conv', 'conv', 'add'], ['conv', 'conv', 'conv', 'conv'], ['add', 'add', 'add', 'add', 'add', 'add'], ['add', 'conv', 'add', 'add', 'add', 'add', 'add', 'add'], ['add', 'add']], 'FLOPs': 4.429, 'acc': 1.47, 'Energy': 0.232, 'Latency': 2.482}]

        for cfg_train in cfg:
            # print(cfg_train)
            model.module.set_active_subnet(cfg_train['resolution'], cfg_train['width'], cfg_train['depth'], cfg_train['kernel_size'], cfg_train['expand_ratio'], cfg_train['type'])
            
            best_acc, is_best, best_epoch = validate(
                train_loader, val_loader, model, criterion, args, 0, logger, logger_curve, best_acc, best_epoch, cfg=cfg_train
            )
        
        exit()
        # ######### reset learning rate #########
        # for group in optimizer.param_groups:
        #     # TODO:
        #     # group['lr'] *= 2
        #     group['lr'] = 0.005
        # print(optimizer)
    else:
        cfg_distill = {'resolution': 32, 'width': [16, 24, 32, 32, 72, 128, 216, 224, 1792], 'depth': [2, 5, 6, 3, 4, 3, 2], 'kernel_size': [5, 5, 5, 5, 5, 3, 5], 'expand_ratio': [1, 6, 6, 5, 5, 6, 6], 'type': [['conv', 'conv'], ['conv', 'conv', 'conv', 'conv', 'conv'], ['conv', 'conv', 'conv', 'conv', 'conv', 'conv'], ['conv', 'conv', 'conv', 'conv', 'conv', 'conv'], ['conv', 'conv', 'conv', 'conv'], ['conv', 'conv', 'conv'], ['conv', 'conv']], 'FLOPs': 6.385, 'acc': 79.34, 'Energy': 0.244, 'Latency': 3.115}

        cfg_train = {'resolution': 32, 'width': [16, 16, 32, 40, 64, 112, 192, 224, 1984], 'depth': [1, 4, 6, 5, 4, 5, 1], 'kernel_size': [3, 5, 5, 5, 5, 5, 5], 'expand_ratio': [1, 4, 6          , 6, 6, 6, 6], 'type': [['conv'], ['conv', 'conv', 'conv', 'conv'], ['conv', 'conv', 'conv', 'conv', 'conv', 'conv'], ['conv', 'conv', 'conv', 'add', 'conv'], ['conv', 'add', 'conv', 'conv'], ['conv', 'add', 'add', 'add', 'add'], ['conv']], 'Energy': 0.211, 'Latency': 2.322, 'FLOPs': 4.635, 'acc': 75.05}
        print(cfg_train)
        model.module.set_active_subnet(cfg_train['resolution'], cfg_train['width'], cfg_train['depth'], cfg_train['kernel_size'], cfg_train['expand_ratio'], cfg_train['type'])

    # TODO:
    # optimizer = torch.optim.SGD(
    #         model.parameters(),
    #         lr=args.optimizer.lr,
    #         momentum=args.optimizer.momentum,
    #         weight_decay=args.weight_decay_weight)
    
    # if args.lr_scheduler.method == 'multistep':
    #     lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler.milestones, gamma=args.lr_scheduler.gamma)
    
    # elif args.lr_scheduler.method == 'cosine':
    #     lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.lr_scheduler.eta_min )

    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)

    # import ipdb
    # ipdb.set_trace()
    drop_connect_only_last_two_stages = getattr(args, 'drop_connect_only_last_two_stages', True)
    model.module.set_dropout_rate(0, 0, drop_connect_only_last_two_stages) 
    # logger.info(args)

    for epoch in range(0, args.epochs):
        # if args.distributed:
        #     train_sampler.set_epoch(epoch)

        args.curr_epoch = epoch
        logger.info('Training lr {}'.format(lr_scheduler.get_lr()[0]))
        
        # train for one epoch
        train_epoch(epoch, model, train_loader, optimizer, criterion, args, logger, logger_curve, cfg_train, cfg_distill, soft_criterion=soft_criterion, lr_scheduler=lr_scheduler)


        # if comm.is_master_process() or args.distributed:
            # validate supernet model
        best_acc, is_best, best_epoch = validate(
            train_loader, val_loader, model, criterion, args,epoch, logger, logger_curve, best_acc, best_epoch, cfg=cfg_train
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
            best_acc,
            0,
            best_epoch, 0,
            is_best = is_best
        )


def train_epoch(
    epoch, 
    model, 
    train_loader, 
    optimizer, 
    criterion, 
    args, logger, logger_curve, cfg_train, cfg_distill, 
    soft_criterion=None, 
    lr_scheduler=None, 
):
    # batch_time = AverageMeter('Time', ':6.3f')
    # data_time = AverageMeter('Data', ':6.3f')
    # losses = AverageMeter('Loss', ':.4e')
    # top1 = AverageMeter('Acc@1', ':6.2f')
    # top5 = AverageMeter('Acc@5', ':6.2f')
    # progress = ProgressMeter(
    #     len(train_loader),
    #     [batch_time, data_time, losses, top1, top5],
    #     prefix="Epoch: [{}]".format(epoch))

    model.train()
    # print('ok')
    num_updates = epoch * len(train_loader)
    
    for batch_idx, (images, target) in enumerate(train_loader):
      
        start_time = time.time()
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        data_time = time.time() - start_time

        optimizer.zero_grad()
        
        model.module.set_active_subnet(cfg_distill['resolution'], cfg_distill['width'], cfg_distill['depth'], cfg_distill['kernel_size'], cfg_distill['expand_ratio'], cfg_distill['type'])
        output = model(images)
        with torch.no_grad():
            soft_logits = output.clone().detach()

        model.module.set_active_subnet(cfg_train['resolution'], cfg_train['width'], cfg_train['depth'], cfg_train['kernel_size'], cfg_train['expand_ratio'], cfg_train['type'])
        output = model(images)
        
        loss = criterion(output, target)
        loss += 5e-3*soft_criterion(output, soft_logits)
        # loss = criterion(output, target)
        
        loss.backward()

        #clip gradients if specfied
        if getattr(args, 'grad_clip_value', None):
            torch.nn.utils.clip_grad_value_(model.parameters(), args.grad_clip_value)

        optimizer.step()
        total_time = time.time() - start_time

        if batch_idx % 20 == 0:
            if not (args.multiprocessing_distributed or args.distributed) or (args.distributed and dist.get_rank() == 0):
                logger.info("[Epoch %d/%d][Step %d/%d] Loss=%.3f Time=%.3f Data Time=%.3f" % (epoch + 1, args.epochs, batch_idx + 1, len(train_loader), loss.item(), total_time, data_time))
                logger_curve.add_scalar('loss/train', loss, epoch*len(train_loader)+batch_idx)
                # break

        # measure elapsed time
        # batch_time.update(time.time() - end)
        # end = time.time()

        num_updates += 1
        # TODO:
        if lr_scheduler is not None:
            lr_scheduler.step()


def validate(
    train_loader, 
    val_loader, 
    model, 
    criterion, 
    args, 
    epoch,
    logger, logger_curve,
    best_acc, best_epoch,
    distributed = False,
    cfg = None
):
    subnets_to_be_evaluated = {
        'attentive_nas_fix_net': cfg
    }

    best_acc, is_best, best_epoch = attentive_nas_eval.validate_infer(
        subnets_to_be_evaluated,
        train_loader,
        val_loader, 
        model, 
        criterion,
        args,
        epoch, best_acc, best_epoch,
        logger_curve,
        logger,
        bn_calibration = True,
    )

    return best_acc, is_best, best_epoch



if __name__ == '__main__':
    main()


