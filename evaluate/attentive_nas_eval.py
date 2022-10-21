# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

import time
from utils.progress import AverageMeter, ProgressMeter, accuracy
import utils.comm as comm

from .imagenet_eval import validate_one_subnet, log_helper

def reduce_tensor(rt, n):
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt

def validate(
    subnets_to_be_evaluated,
    train_loader, 
    val_loader, 
    model, 
    criterion, 
    args, 
    epoch, best_acc_min, best_acc_max, best_acc_random, best_epoch, 
    logger_curve,
    logger,
    bn_calibration=True,
):
    # supernet = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    # print('supernet',isinstance(model, torch.nn.parallel.DistributedDataParallel))
    is_best = False
    supernet = model.module

    results = []
    top1_list, top5_list = [],  []
    best_epoch_min = 0
    best_epoch_max = 0
    # best_acc_random = 0
    with torch.no_grad():
        for net_id in subnets_to_be_evaluated:
            if net_id == 'attentive_nas_min_net': 
                supernet.sample_min_subnet()
                print(supernet.sample_min_subnet())
            elif net_id == 'attentive_nas_max_net':
                supernet.sample_max_subnet()
                print(supernet.sample_max_subnet())
            elif net_id.startswith('attentive_nas_random_net'):
                supernet.sample_active_subnet()
                print(supernet.sample_active_subnet())
            else:
                supernet.set_active_subnet(
                    subnets_to_be_evaluated[net_id]['resolution'],
                    subnets_to_be_evaluated[net_id]['width'],
                    subnets_to_be_evaluated[net_id]['depth'],
                    subnets_to_be_evaluated[net_id]['kernel_size'],
                    subnets_to_be_evaluated[net_id]['expand_ratio'],
                    subnets_to_be_evaluated[net_id]['type']
                )

            subnet = supernet.get_active_subnet()
            subnet_cfg = supernet.get_active_subnet_settings()
            # print(subnet_cfg)
            subnet.cuda()

            if bn_calibration:
                subnet.eval()
                subnet.reset_running_stats_for_calibration()

                # estimate running mean and running statistics
                logger.info('Calirating bn running statistics')
                for batch_idx, (images, _) in enumerate(train_loader):
                    if batch_idx >= args.post_bn_calibration_batch_num:
                        break
                    if getattr(args, 'use_clean_images_for_subnet_training', False):
                        _, images = images
                    images = images.cuda(non_blocking=True)
                    subnet(images)  #forward only

            acc1, acc5 = validate_one_subnet(
                val_loader, subnet, criterion, args, logger
            )
            
            if args.distributed:
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            
            if net_id == 'attentive_nas_min_net':
                if acc1 > best_acc_min:
                    best_acc_min = acc1
                    best_epoch_min = epoch
            elif net_id == 'attentive_nas_max_net':
                if acc1 > best_acc_max:
                    best_acc_max = acc1
                    best_epoch_max = epoch
                    is_best = True
            elif net_id == 'attentive_nas_random_net':
                if acc1 > best_acc_random:
                    best_acc_random = acc1
                    # best_epoch_min = epoch
           
            if not (args.multiprocessing_distributed or args.distributed) or (args.distributed and dist.get_rank() == 0):
                if net_id == 'attentive_nas_min_net':
                    logger_curve.add_scalar('min_net_acc/val', acc1, epoch)
                    # logger_curve.add_scalar('min_net_acc/val', acc5, epoch)
                    logger.info("min net Epoch:%d Acc1:%.3f (Best Acc1:%.3f) Acc5:%.3f Best Epoch:%d" % (epoch, acc1, best_acc_min, acc5, best_epoch_min))
                elif net_id == 'attentive_nas_max_net':
                    logger_curve.add_scalar('max_net_acc/val', acc1, epoch)
                    # logger_curve.add_scalar('max_net_acc/val', acc5, epoch)
                    logger.info("max net Epoch:%d Acc1:%.3f (Best Acc1:%.3f) Acc5:%.3f Best Epoch:%d" % (epoch, acc1, best_acc_max, acc5, best_epoch_max))
                elif net_id == 'attentive_nas_random_net':
                    logger_curve.add_scalar('random_net_acc/val', acc1, epoch)
                    # logger_curve.add_scalar('random_net_acc/val', acc5, epoch)
                    logger.info("random net Epoch:%d Acc1:%.3f (Best Acc1:%.3f) Acc5:%.3f" % (epoch, acc1, best_acc_random, acc5))

    return best_acc_min, best_acc_max, best_acc_random, is_best, best_epoch_min, best_epoch_max


def validate_infer(
    subnets_to_be_evaluated,
    train_loader, 
    val_loader, 
    model, 
    criterion, 
    args, 
    epoch, best_acc, best_epoch, 
    logger_curve,
    logger,
    bn_calibration=True,
):
    # supernet = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    # print('supernet',isinstance(model, torch.nn.parallel.DistributedDataParallel))
    is_best = False
    supernet = model.module

    results = []
    top1_list, top5_list = [],  []
    best_epoch_min = 0
    best_epoch_max = 0
    # best_acc_random = 0
    with torch.no_grad():
        for net_id in subnets_to_be_evaluated:
            supernet.set_active_subnet(
                subnets_to_be_evaluated[net_id]['resolution'],
                subnets_to_be_evaluated[net_id]['width'],
                subnets_to_be_evaluated[net_id]['depth'],
                subnets_to_be_evaluated[net_id]['kernel_size'],
                subnets_to_be_evaluated[net_id]['expand_ratio'],
                subnets_to_be_evaluated[net_id]['type']
            )

            subnet = supernet.get_active_subnet()
            subnet_cfg = supernet.get_active_subnet_settings()
            subnet.cuda()

            if bn_calibration:
                subnet.eval()
                subnet.reset_running_stats_for_calibration()

                # estimate running mean and running statistics
                # logger.info('Calirating bn running statistics')
                for batch_idx, (images, _) in enumerate(train_loader):
                    if batch_idx >= args.post_bn_calibration_batch_num:
                        break
                    if getattr(args, 'use_clean_images_for_subnet_training', False):
                        _, images = images
                    images = images.cuda(non_blocking=True)
                    subnet(images)  #forward only

            acc1, acc5 = validate_one_subnet(
                val_loader, subnet, criterion, args, logger
            )
            
            if args.distributed:
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            
            if acc1 > best_acc:
                best_acc = acc1
            if not (args.multiprocessing_distributed or args.distributed) or (args.distributed and dist.get_rank() == 0):
                logger_curve.add_scalar('acc/val', acc1, epoch)
                # logger_curve.add_scalar('random_net_acc/val', acc5, epoch)
                logger.info("Epoch:%d Acc1:%.3f (Best Acc1:%.3f) Acc5:%.3f" % (epoch, acc1, best_acc, acc5))

    return best_acc, is_best, best_epoch



def validate_collect(
    subnets_to_be_evaluated,
    train_loader, 
    val_loader, 
    model, 
    criterion, 
    args, 
    logger,
    bn_calibration=True,
):
    # supernet = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    # print('supernet',isinstance(model, torch.nn.parallel.DistributedDataParallel))
    # is_best = False
    supernet = model.module


    with torch.no_grad():
        for net_id in subnets_to_be_evaluated:
            supernet.set_active_subnet(
                subnets_to_be_evaluated[net_id]['resolution'],
                subnets_to_be_evaluated[net_id]['width'],
                subnets_to_be_evaluated[net_id]['depth'],
                subnets_to_be_evaluated[net_id]['kernel_size'],
                subnets_to_be_evaluated[net_id]['expand_ratio'],
                subnets_to_be_evaluated[net_id]['type']
            )

            subnet = supernet.get_active_subnet()
            # subnet_cfg = supernet.get_active_subnet_settings()
            subnet.cuda()

            if bn_calibration:
                subnet.eval()
                subnet.reset_running_stats_for_calibration()

                # estimate running mean and running statistics
                # logger.info('Calirating bn running statistics')
                for batch_idx, (images, _) in enumerate(train_loader):
                    if batch_idx >= args.post_bn_calibration_batch_num:
                        break
                    if getattr(args, 'use_clean_images_for_subnet_training', False):
                        _, images = images
                    images = images.cuda(non_blocking=True)
                    subnet(images)  #forward only

            acc1, acc5 = validate_one_subnet(
                val_loader, subnet, criterion, args, logger
            )
            
            if args.distributed:
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            
            # if acc1 > best_acc:
            #     best_acc = acc1
            # if not (args.multiprocessing_distributed or args.distributed) or (args.distributed and dist.get_rank() == 0):
            #     logger_curve.add_scalar('acc/val', acc1, epoch)
            #     # logger_curve.add_scalar('random_net_acc/val', acc5, epoch)
            #     logger.info("Epoch:%d Acc1:%.3f (Best Acc1:%.3f) Acc5:%.3f" % (epoch, acc1, best_acc, acc5))

    return round(acc1.item(),2)