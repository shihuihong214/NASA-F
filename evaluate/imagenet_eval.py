# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
# import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import time

from utils.progress import AverageMeter, ProgressMeter, accuracy
from utils.flops_counter import count_net_flops_and_params
import models


def log_helper(summary, logger=None):
    if logger:
        logger.info(summary)
    else:
        print(summary)


def validate_one_subnet(
    val_loader,
    subnet,
    criterion,
    args, 
    logger=None, 
):
    # batch_time = AverageMeter('Time', ':6.3f')
    # losses = AverageMeter('Loss', ':.4e')
    # top1 = AverageMeter('Acc@1', ':6.2f')
    # top5 = AverageMeter('Acc@5', ':6.2f')
    # progress = ProgressMeter(
    #             len(val_loader),
    #             [batch_time, losses, top1, top5],
    #             prefix='Test: ')

    log_helper('evaluating...', logger)   
    #evaluation
    end = time.time()

    subnet.cuda()
    subnet.eval() # freeze again all running stats
    prec1_list = []
    prec5_list = []
   
    for batch_idx, (images, target) in enumerate(val_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = subnet(images)
        loss = criterion(output, target).item()

        # measure accuracy 
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        prec1_list.append(prec1)
        prec5_list.append(prec5)
    acc1 = sum(prec1_list)/len(prec1_list)
    acc5 = sum(prec5_list)/len(prec5_list)    

    # compute flops
    if getattr(subnet, 'module', None):
        resolution = subnet.module.resolution
    else:
        resolution = subnet.resolution
    data_shape = (1, 3, resolution, resolution)

    # flops, params = count_net_flops_and_params(subnet, data_shape)
    return acc1, acc5


