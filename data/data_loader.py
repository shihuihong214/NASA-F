# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from __future__ import print_function

import torch
import torchvision.transforms.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import math
import sys
import random
from PIL import Image
import torchvision.datasets as dset

from torch.utils.data.distributed import DistributedSampler
import os
from .autoaugment import CIFAR10Policy
from .data_transform import get_data_transform

def build_data_loader(args):
    if args.dataset == 'imagenet':
        return build_default_imagenet_data_loader(args)
    elif args.dataset == 'cifar10' or args.dataset == 'cifar100':
        return build_default_cifar_data_loader(args)
    else:
        raise NotImplementedError
    
def build_default_imagenet_data_loader(args):
    traindir = os.path.join(args.dataset_dir, "train")
    valdir = os.path.join(args.dataset_dir, "val")

    #build transforms
    train_transform = get_data_transform(args, is_training=True, augment=args.augment)
    test_transform = get_data_transform(args, is_training=False, augment=args.augment)

    #build datasets
    if not getattr(args, 'data_loader_cross_validation', False):
        train_dataset = datasets.ImageFolder(traindir, train_transform)
        val_dataset = datasets.ImageFolder(valdir, test_transform)
    #else:
    #    my_dataset = datasets.ImageFolder(traindir)
    #    train_dataset, val_dataset = torch.utils.data.random_split(
    #        my_dataset, [args.data_split_ntrain, args.data_split_nval], generator=torch.Generator().manual_seed(args.data_split_seed)
    #    )
    #    train_dataset = MyDataset( train_dataset, train_transform)
    #    val_dataset = MyDataset(val_dataset, test_transform)


    #build data loaders
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last = getattr(args, 'drop_last', True),
        num_workers=args.data_loader_workers_per_gpu,
        pin_memory=True,
    )

    if args.distributed and getattr(args, 'distributed_val', True):
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        val_sampler = None

    eval_batch_size = min(args.batch_size, 16) \
        if not getattr(args, 'eval_only', False) else args.batch_size

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=args.data_loader_workers_per_gpu,
        drop_last=False,
        pin_memory=True,
        sampler=val_sampler,
    )

    return train_loader, val_loader, train_sampler


def build_default_cifar_data_loader(args):
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2023, 0.1994, 0.2010)),
    ])

    if args.dataset == 'cifar10':
        train_dataset = dset.CIFAR10(root=args.dataset_dir, train=True, download=False, transform=transform_train)
        val_dataset = dset.CIFAR10(root=args.dataset_dir, train=False, download=False, transform=transform_test)
    elif args.dataset == 'cifar100':
        train_dataset = dset.CIFAR100(root=args.dataset_dir, train=True, download=False, transform=transform_train)
        val_dataset = dset.CIFAR100(root=args.dataset_dir, train=False, download=False, transform=transform_test)


    #build data loaders
    if args.distributed:
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last = getattr(args, 'drop_last', True),
        num_workers=args.data_loader_workers_per_gpu,
        pin_memory=True,
    )

    if args.distributed and getattr(args, 'distributed_val', True):
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        val_sampler = None

    eval_batch_size = min(args.batch_size, 16) \
        if not getattr(args, 'eval_only', False) else args.batch_size

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=args.data_loader_workers_per_gpu,
        drop_last=False,
        pin_memory=True,
        sampler=val_sampler,
    )

    return train_loader, val_loader, train_sampler


