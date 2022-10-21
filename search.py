# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import random

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
import models
# from plot import Latency
from utils.config import setup
import utils.comm as comm
import utils.saver as saver

from data.data_loader import build_data_loader
from evaluate import attentive_nas_eval as attentive_nas_eval
import utils.logging as logging
import argparse
import os
import heapq

"""
    using multiple nodes to run evolutionary search:
    1) each GPU will evaluate its own sub-networks
    2) all evaluation results will be aggregated on GPU 0
"""
parser = argparse.ArgumentParser(description='Test AlphaNet Models')
parser.add_argument('--config-file', default='./configs/parallel_supernet_evo_search.yml')
parser.add_argument('--machine-rank', default=0, type=int, 
                    help='machine rank, distributed setting')
parser.add_argument('--num-machines', default=1, type=int, 
                    help='number of nodes, distributed setting')
parser.add_argument('--dist-url', default="tcp://127.0.0.1:10001", type=str, 
                    help='init method, distributed setting')
parser.add_argument('--seed', default=1, type=int, 
                    help='default random seed')
parser.add_argument("--port", type=str, default="10001")
parser.add_argument('--gpu', type=str, default='0', 
                    help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument("--quant", action='store_true', default=False)
run_args = parser.parse_args()


logger = logging.get_logger(__name__)

def eval_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu  # local rank, local machine cuda id
    args.local_rank = args.gpu
    args.batch_size = args.batch_size_per_gpu

    # global_rank = args.gpu + args.machine_rank * ngpus_per_node
    # dist.init_process_group(
    #     backend=args.dist_backend, 
    #     init_method=args.dist_url,
    #     world_size=args.world_size, 
    #     rank=global_rank
    # )
    if args.distributed:
        os.environ['MASTER_PORT'] = args.port
        dist.init_process_group(backend="nccl")

    # Setup logging format.
    logging.setup_logging("stdout.log", 'w')

    # synchronize is needed here to prevent a possible timeout after calling
    # init_process_group
    # See: https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
    # comm.synchronize()

    # args.rank = comm.get_rank() # global rank
    # torch.cuda.set_device(args.gpu)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # build the supernet
    logger.info("=> creating model '{}'".format(args.arch))
    model = models.model_factory.create_model(args)
    # model.cuda(args.gpu)
    # model = comm.get_parallel_model(model, args.gpu) #local rank
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

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().cuda()

    ## load dataset, train_sampler: distributed
    train_loader, val_loader, train_sampler =  build_data_loader(args)

    assert args.resume
    #reloading model
    # model.module.load_weights_from_pretrained_models(args.resume)

    # TODO: collect the corresponding relationship between FLOPs and Energy
    # FLOPs = []
    # EDP = []
    # Energy_min = []
    # Energy = []
    # Energy_wo = []
    # Latency_min = []
    # Latency = []
    # Buffer = []
    # for iter in range(200):
    #     cfg = model.module._sample_active_subnet() 
    #     print('cfg', cfg)
    #     print('===================== {} ====================='.format(iter))
    #     edp, min_energy, energy, energy_wodram, min_latency, latency, flops, buffer = model.module.compute_EDP(cfg, stage='fine')
    #     print('Energy', min_energy)
    #     print('Energy w/o DRAM', energy_wodram)
    #     print('FLOPs', flops)
    #     print('Model size', buffer)
        
    #     EDP.append(edp)
    #     Energy_min.append(min_energy)
    #     Energy.append(energy)
    #     Energy_wo.append(energy_wodram)
    #     Latency_min.append(min_latency)
    #     Latency.append(latency)
    #     FLOPs.append(flops)
    #     Buffer.append(buffer)
    # print('EDP =', EDP)
    # print('Energy_min =', Energy_min)
    # print('Energy =', Energy)
    # print('Energy_wodram =', Energy_wo)
    # print('Latency_min =', Latency_min)
    # print('Latency =', Latency)
    # print('FLOPs =', FLOPs)
    # print('Buffer =', Buffer)
    # exit()

    ################# computational cost vs EDP ###############
    # Computational_Cost = []
    # EDP = []
    # for iter in range(200):
    #     cfg = model.module._sample_active_subnet() 
    #     print('cfg', cfg)
    #     print('===================== {} ====================='.format(iter))
    #     edp, computation_cost = model.module.compute_EDP(cfg, stage='fine', Conv_PE=168)
    #     print('EDP', edp)
    #     print('Computational_Cost', computation_cost)
       
    #     EDP.append(edp)
    #     Computational_Cost.append(computation_cost)
        
    # print('EDP =', EDP)
    # print('Computational_Cost =', Computational_Cost)
    # exit()
    # new_candidates = []
    # candidates = [{'resolution': 32, 'width': [24, 24, 32, 40, 64, 120, 192, 224, 1984], 'depth': [1, 5, 5, 5, 4, 3, 2], 'kernel_size': [3, 5, 5, 3, 5, 5, 3], 'expand_ratio': [1, 5, 6, 4, 4, 6, 6], 'type': [['conv'], ['conv', 'conv', 'conv', 'conv', 'conv'], ['conv', 'conv', 'conv', 'conv', 'add'], ['conv', 'conv', 'conv', 'add', 'add'], ['conv', 'add', 'add', 'conv'], ['conv', 'conv', 'add'], ['conv', 'add']], 'Energy': 0.193, 'Latency': 2.471, 'FLOPs': 4.83, 'acc': 69.3}, {'resolution': 32, 'width': [24, 24, 32, 40, 64, 120, 192, 224, 1792], 'depth': [1, 5, 5, 5, 4, 3, 2], 'kernel_size': [3, 5, 5, 3, 3, 5, 3], 'expand_ratio': [1, 5, 6, 6, 4, 6, 6], 'type': [['conv'], ['conv', 'conv', 'conv', 'conv', 'conv'], ['conv', 'conv', 'conv', 'conv', 'add'], ['conv', 'conv', 'conv', 'add', 'add'], ['conv', 'conv', 'add', 'conv'], ['conv', 'add', 'conv'], ['conv', 'add']], 'Energy': 0.198, 'Latency': 2.415, 'FLOPs': 5.014, 'acc': 69.25}, {'resolution': 32, 'width': [24, 16, 32, 40, 64, 120, 192, 216, 1792], 'depth': [1, 5, 6, 3, 4, 3, 1], 'kernel_size': [3, 5, 5, 3, 5, 5, 3], 'expand_ratio': [1, 5, 6, 4, 4, 6, 6], 'type': [['conv'], ['conv', 'conv', 'conv', 'conv', 'conv'], ['conv', 'conv', 'conv', 'conv', 'add', 'conv'], ['conv', 'conv', 'conv'], ['conv', 'add', 'add', 'conv'], ['conv', 'add', 'add'], ['conv']], 'Energy': 0.177, 'Latency': 2.247, 'FLOPs': 4.382, 'acc': 69.16}, {'resolution': 32, 'width': [24, 16, 32, 40, 64, 120, 192, 216, 1792], 'depth': [2, 5, 6, 3, 4, 3, 1], 'kernel_size': [3, 5, 5, 3, 5, 5, 3], 'expand_ratio': [1, 5, 6, 4, 4, 6, 6], 'type': [['conv', 'add'], ['conv', 'conv', 'conv', 'conv', 'conv'], ['conv', 'conv', 'conv', 'conv', 'add', 'conv'], ['conv', 'conv', 'conv'], ['conv', 'add', 'add', 'conv'], ['conv', 'add', 'add'], ['conv']], 'Energy': 0.178, 'Latency': 2.305, 'FLOPs': 4.407, 'acc': 69.14}, {'resolution': 32, 'width': [24, 16, 32, 40, 64, 120, 192, 216, 1792], 'depth': [2, 5, 6, 4, 4, 3, 1], 'kernel_size': [3, 5, 5, 3, 5, 5, 3], 'expand_ratio': [1, 5, 6, 6, 4, 6, 6], 'type': [['conv', 'add'], ['conv', 'conv', 'conv', 'conv', 'conv'], ['conv', 'conv', 'conv', 'conv', 'add', 'conv'], ['conv', 'conv', 'conv', 'add'], ['conv', 'add', 'add', 'conv'], ['conv', 'add', 'conv'], ['conv']], 'FLOPs': 4.662, 'acc': 69.12, 'Energy': 0.185, 'Latency': 2.084}, {'resolution': 32, 'width': [24, 16, 32, 40, 64, 120, 192, 224, 1984], 'depth': [1, 5, 5, 5, 5, 3, 1], 'kernel_size': [3, 5, 5, 3, 5, 3, 3], 'expand_ratio': [1, 5, 6, 4, 4, 6, 6], 'type': [['conv'], ['conv', 'conv', 'conv', 'conv', 'conv'], ['conv', 'conv', 'conv', 'conv', 'add'], ['conv', 'conv', 'conv', 'add', 'add'], ['conv', 'add', 'add', 'add', 'add'], ['conv', 'add', 'add'], ['conv']], 'Energy': 0.175, 'Latency': 2.012, 'FLOPs': 4.277, 'acc': 68.98}, {'resolution': 32, 'width': [24, 16, 32, 40, 64, 120, 192, 224, 1792], 'depth': [2, 5, 5, 4, 4, 3, 1], 'kernel_size': [3, 5, 5, 3, 5, 5, 3], 'expand_ratio': [1, 5, 6, 4, 4, 6, 6], 'type': [['conv', 'add'], ['conv', 'conv', 'conv', 'conv', 'conv'], ['conv', 'conv', 'conv', 'conv', 'add'], ['conv', 'conv', 'conv', 'add'], ['conv', 'add', 'add', 'conv'], ['conv', 'add', 'add'], ['conv']], 'Energy': 0.17, 'Latency': 1.948, 'FLOPs': 4.27, 'acc': 68.93}, {'resolution': 32, 'width': [24, 24, 32, 40, 64, 120, 192, 224, 1792], 'depth': [2, 5, 3, 4, 4, 3, 1], 'kernel_size': [3, 5, 5, 3, 5, 5, 3], 'expand_ratio': [1, 5, 6, 6, 4, 6, 6], 'type': [['conv', 'add'], ['conv', 'conv', 'conv', 'conv', 'add'], ['conv', 'conv', 'conv'], ['conv', 'conv', 'conv', 'add'], ['conv', 'add', 'add', 'conv'], ['conv', 'add', 'conv'], ['conv']], 'Energy': 0.164, 'Latency': 1.81, 'FLOPs': 4.331, 'acc': 68.33}, {'resolution': 32, 'width': [24, 16, 32, 40, 64, 120, 192, 224, 1792], 'depth': [1, 5, 3, 3, 4, 3, 1], 'kernel_size': [3, 5, 5, 3, 5, 5, 5], 'expand_ratio': [1, 5, 6, 4, 4, 6, 6], 'type': [['conv'], ['conv', 'conv', 'conv', 'conv', 'add'], ['conv', 'conv', 'conv'], ['conv', 'conv', 'conv'], ['conv', 'add', 'add', 'conv'], ['conv', 'add', 'conv'], ['conv']], 'Energy': 0.152, 'Latency': 1.716, 'FLOPs': 3.901, 'acc': 67.91}, {'resolution': 32, 'width': [24, 16, 32, 40, 64, 120, 192, 224, 1984], 'depth': [1, 5, 3, 5, 5, 3, 1], 'kernel_size': [3, 5, 5, 3, 5, 5, 5], 'expand_ratio': [1, 4, 6, 4, 4, 6, 6], 'type': [['conv'], ['conv', 'conv', 'conv', 'conv', 'add'], ['conv', 'conv', 'conv'], ['conv', 'conv', 'conv', 'conv', 'conv'], ['conv', 'add', 'add', 'add', 'add'], ['conv', 'add', 'conv'], ['conv']], 'FLOPs': 3.708, 'acc': 67.27, 'Energy': 0.153, 'Latency': 1.611}, {'resolution': 32, 'width': [24, 24, 32, 40, 64, 112, 192, 224, 1792], 'depth': [1, 5, 3, 3, 5, 8, 2], 'kernel_size': [3, 5, 5, 3, 5, 3, 5], 'expand_ratio': [1, 4, 6, 5, 6, 6, 6], 'type': [['conv'], ['conv', 'conv', 'conv', 'add', 'conv'], ['conv', 'conv', 'conv'], ['conv', 'conv', 'conv'], ['conv', 'conv', 'add', 'add', 'add'], ['conv', 'add', 'conv', 'add', 'add', 'add', 'add', 'add'], ['conv', 'add']], 'FLOPs': 4.994, 'acc': 66.67, 'Energy': 0.244, 'Latency': 2.657}, {'resolution': 32, 'width': [24, 16, 24, 40, 64, 120, 192, 216, 1984], 'depth': [1, 5, 3, 5, 5, 3, 1], 'kernel_size': [3, 5, 5, 3, 3, 5, 3], 'expand_ratio': [1, 4, 6, 4, 4, 6, 6], 'type': [['conv'], ['conv', 'conv', 'conv', 'conv', 'conv'], ['conv', 'conv', 'conv'], ['conv', 'conv', 'add', 'add', 'add'], ['conv', 'add', 'add', 'conv', 'add'], ['conv', 'add', 'add'], ['conv']], 'Energy': 0.135, 'Latency': 1.541, 'FLOPs': 3.133, 'acc': 65.94}, {'resolution': 32, 'width': [24, 16, 24, 40, 64, 120, 192, 216, 1984], 'depth': [1, 3, 3, 5, 5, 3, 1], 'kernel_size': [3, 5, 5, 3, 5, 3, 5], 'expand_ratio': [1, 4, 6, 6, 4, 6, 6], 'type': [['conv'], ['conv', 'conv', 'conv'], ['conv', 'conv', 'conv'], ['conv', 'conv', 'add', 'add', 'conv'], ['conv', 'conv', 'add', 'conv', 'add'], ['conv', 'add', 'add'], ['conv']], 'FLOPs': 3.034, 'acc': 64.44, 'Energy': 0.134, 'Latency': 1.447}]
    
    # new_candidates.append(candidates[0])
    # for cfg in candidates:
    #     if not (cfg['Latency'] < new_candidates[-1]['Latency']):
    #         pass
    #     else:
    #         new_candidates.append(cfg) 
    # print(new_candidates)
    
    # Acc = []
    # Ratio = []
    # for cfg in new_candidates:
    #     Acc.append(cfg['acc'])
    #     Ratio.append(model.module.compute_EDP(cfg, stage='coarse', Conv_PE=168))
    # print(Acc)
    # print(Ratio)
    # exit()

    saver.load_search_checkpoints(args, model)

    # FIXME: test ckpt
    # cfg = model.module._sample_active_subnet()
    # print(cfg)
    # acc = attentive_nas_eval.validate_collect(
    #         {'test': cfg},
    #         train_loader,
    #         val_loader,
    #         model,
    #         criterion, 
    #         args, 
    #         logger,)
    # print(acc)
    # exit()

    # if train_sampler:
    #     train_sampler.set_epoch(0)
    
    # targeted_min_flops = args.evo_search.targeted_min_flops
    # targeted_max_flops = args.evo_search.targeted_max_flops
    # targeted_min_EDP = args.evo_search.targeted_min_EDP
    # targeted_max_EDP = args.evo_search.targeted_max_EDP
    targeted_min_FLOPs = args.evo_search.targeted_min_FLOPs
    targeted_max_FLOPs = args.evo_search.targeted_max_FLOPs
    # targeted_min_Latency = args.evo_search.targeted_min_Latency
    # targeted_max_Latency = args.evo_search.targeted_max_Latency
    # targeted_min_Energy = args.evo_search.targeted_min_Energy
    # targeted_max_Energy = args.evo_search.targeted_max_Energy
    targeted_min_EDP = args.evo_search.targeted_min_EDP
    targeted_max_EDP = args.evo_search.targeted_max_EDP
    coarse_iter = args.evo_search.coarse_iter
    # print(targeted_min_FLOPs)
    # print(int(targeted_min_FLOPs))
    # targeted_min_Acc = args.evo_search.targeted_min_Acc
    

    # run evolutionary search
    parent_popu = []
    model.module.settings_for_mixture_training(pretrain_conv=False, pretrain_add=False, mix=True, mix_training=True, prob=0.5)
    for idx in range(args.evo_search.parent_popu_size):
        # print('ok')
        # if idx == 0:
        #     cfg = model.module.sample_min_subnet()
        #     # cfg = acc_constraint(cfg)
        #     # cfg['EDP'] = model.module.compute_active_subnet_flops()
        #     # cfg['FLOPs'] = model.module.compute_EDP(cfg, stage='coarse')
        #     cfg['Energy'], cfg['Latency'], cfg['FLOPs'] = model.module.compute_EDP(cfg, stage ='fine')
        # else:
            # cfg = model.module.sample_active_subnet_within_range(
            #     targeted_min_EDP, targeted_max_EDP
            # )
        while True:
            cfg = model.module._sample_active_subnet()
            # print(cfg)
            # cfg = acc_constraint(cfg)
            cfg['FLOPs'] = model.module.compute_EDP(cfg, stage='coarse', Conv_PE=168)
            if (cfg['FLOPs'] >= targeted_min_FLOPs and cfg['FLOPs'] <= targeted_max_FLOPs):
                # TODO:
                # cfg['EDP'], cfg['FLOPs'] = model.module.compute_EDP(cfg, stage ='fine', Conv_PE=168)
                break
        
        cfg['net_id'] = 'parent_'+str(idx)
        # print('cfg',cfg)
        cfg['acc'] = attentive_nas_eval.validate_collect(
            {'test': cfg},
            train_loader,
            val_loader,
            model,
            criterion, 
            args, 
            logger,)
        # cfg['acc'] = random.random()
        parent_popu.append(cfg)

    pareto_global = {}
    for cfg in parent_popu:
        # f = round(cfg['FLOPs']/5,2)*5
        f = round(cfg['FLOPs'],1)
        if f not in pareto_global or pareto_global[f]['acc'] < cfg['acc']:
            pareto_global[f] = cfg
        # pareto_global[cfg['FLOPs']] = cfg
    
    for evo in range(args.evo_search.evo_iter):   

        # next batch of sub-networks to be evaluated
        parent_popu = []
        # mutate 
        for idx in range(args.evo_search.mutate_size):
            while True:
                old_cfg = random.choice(list(pareto_global.values()))
                cfg = model.module.mutate_and_reset(old_cfg, prob=args.evo_search.mutate_prob)
                # cfg['EDP'] = model.module.compute_active_subnet_flops()
                # cfg = acc_constraint(cfg)
                if evo < coarse_iter:
                    cfg['FLOPs'] = model.module.compute_EDP(cfg, stage='coarse')
                    if (cfg['FLOPs'] >= targeted_min_FLOPs and cfg['FLOPs'] <= targeted_max_FLOPs):
                        break
                else:
                    cfg['EDP'], cfg['FLOPs'] = model.module.compute_EDP(cfg, stage ='fine')
                    # if args.metric == "Latency":
                    #     if (cfg['Latency'] >= targeted_min_Latency and cfg['Latency'] <= targeted_max_Latency):
                    #         break
                    # elif args.metric == "Energy":
                    #     if (cfg['Energy'] >= targeted_min_Energy and cfg['Energy'] <= targeted_max_Energy):
                    #         break
                    if (cfg['EDP'] >= targeted_min_EDP and cfg['EDP'] <= targeted_max_EDP):
                            break
                    # else:
                    #     print('Wrong hardware metrics !!!!!!')
                    #     exit()
            cfg['acc'] = attentive_nas_eval.validate_collect(
            {'test': cfg},
            train_loader,
            val_loader,
            model,
            criterion, 
            args, 
            logger,)
            # cfg['acc'] = random.random()
            parent_popu.append(cfg)

        # cross over
        for idx in range(args.evo_search.crossover_size):
            while True:
                cfg1 = random.choice(list(pareto_global.values()))
                cfg2 = random.choice(list(pareto_global.values()))
                cfg = model.module.crossover_and_reset(cfg1, cfg2)
                
                if evo < coarse_iter:
                    cfg['FLOPs'] = model.module.compute_EDP(cfg, stage='coarse')
                    if (cfg['FLOPs'] >= targeted_min_FLOPs and cfg['FLOPs'] <= targeted_max_FLOPs):
                        break
                else:
                    # print('ok')
                    cfg['EDP'], cfg['FLOPs'] = model.module.compute_EDP(cfg, stage ='fine')
                    # if args.metric == "Latency":
                    #     if (cfg['Latency'] >= targeted_min_Latency and cfg['Latency'] <= targeted_max_Latency):
                    #         break
                    # elif args.metric == "Energy":
                    #     if (cfg['Energy'] >= targeted_min_Energy and cfg['Energy'] <= targeted_max_Energy):
                    #         break
                    if (cfg['EDP'] >= targeted_min_EDP and cfg['EDP'] <= targeted_max_EDP):
                            break
                    # else:
                    #     print('Wrong hardware metrics !!!!!!')
                    #     exit()
                    # # if (cfg['Latency'] >= targeted_min_Latency and cfg['Latency'] <= targeted_max_Latency):
                    # if (cfg['Energy'] >= targeted_min_Energy and cfg['Energy'] <= targeted_max_Energy):
                    #     break
            cfg['acc'] = attentive_nas_eval.validate_collect(
            {'test': cfg},
            train_loader,
            val_loader,
            model,
            criterion, 
            args, 
            logger,)
            # cfg['acc'] = random.random()
            parent_popu.append(cfg)
    

        if evo == coarse_iter:
            args.evo_search.mutate_size = 30
            args.evo_search.crossover_size = 30
            pareto_list = list(pareto_global.values())
            pareto_global = {}
            for cfg in pareto_list:
                cfg['EDP'], cfg['FLOPs'] = model.module.compute_EDP(cfg, stage ='fine')
                # if args.metric == "Latency":
                #     # f = round(cfg['Latency']/5,2)*5
                #     f = round(cfg['Latency'],1)
                # elif args.metric == "Energy":
                #     # f = round(cfg['Energy']/5,3)*5
                #     f = round(cfg['Energy'],2)
                f = round(cfg['EDP'],1)
                if f not in pareto_global or pareto_global[f]['acc'] < cfg['acc']:
                    pareto_global[f] = cfg
                # pareto_global[cfg['Latency']] = cfg
                # pareto_global[cfg['Energy']] = cfg

        # update the Pareto frontier
        # in this case, we search the best FLOPs vs. accuracy trade-offs
        # for cfg in parent_popu:
        #     Acc = round(cfg['Acc'],1)
        #     if Acc not in pareto_global or pareto_global[Acc]['Latency'] > cfg['Latency'] or pareto_global[Acc]['EDP'] > cfg['EDP']:
        #         pareto_global[Acc] = cfg
        
        for cfg in parent_popu:
            if evo < coarse_iter:
                # f = round(cfg['FLOPs']/5,2)*5
                f = round(cfg['FLOPs'],1)
                if f not in pareto_global or pareto_global[f]['acc'] < cfg['acc']:
                    pareto_global[f] = cfg
            else:
                # if args.metric == "Latency":
                #     # Latency = round(cfg['Latency']/5,2)*5
                #     Latency = round(cfg['Latency'],1)
                #     if Latency not in pareto_global or pareto_global[Latency]['acc'] < cfg['acc']:
                #         pareto_global[Latency] = cfg
                # elif args.metric == "Energy":
                #     # Energy = round(cfg['Energy']/5,3)*5
                #     Energy = round(cfg['Energy'],2)
                #     if Energy not in pareto_global or pareto_global[Energy]['acc'] < cfg['acc']:
                #         pareto_global[Energy] = cfg
                f = round(cfg['EDP'],1)
                if f not in pareto_global or pareto_global[f]['acc'] < cfg['acc']:
                    pareto_global[f] = cfg


        # upodate top-k
        # import ipdb
        # ipdb.set_trace()
        topk = list(pareto_global.values())
        # print('pareto_global',pareto_global)
        # print('topk',topk)
        topk.sort(key=lambda x: x['acc'], reverse=True)
        # print(topk[0]['EDP'])
        print("===========================Epoch {}===========================".format(evo))
        print(topk)
    
    # TODO:
    acc = []
    # EDP = []
    FLOPs = []
    # Energy = []
    # Latency = []
    for candidate in topk:
        acc.append(candidate['acc'])
        # if args.metric == "Latency":
        #     Latency.append(candidate['Latency'])
        # elif args.metric == "Energy":
        #     Energy.append(candidate['Energy'])
        # EDP.append(candidate['EDP'])
        FLOPs.append(candidate['FLOPs'])
        
    print('')
    print('acc =', acc)
    # if args.metric == "Latency":
    #     print('Latency =', Latency)
    # elif args.metric == "Energy":
    #     print('Energy =', Energy)
    # print('EDP =', EDP)
    print('FLOPs =', FLOPs)


if __name__ == '__main__':
    # setup enviroments
    args = setup(run_args.config_file)
    args.dist_url = run_args.dist_url
    args.machine_rank = run_args.machine_rank
    args.num_nodes = run_args.num_machines
    args.port = run_args.port
    args.gpu = run_args.gpu

    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.num_nodes
        assert args.world_size > 1, "only support DDP settings"
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # eval_worker process function
        mp.spawn(eval_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # raise NotImplementedError
        eval_worker(args.gpu, ngpus_per_node, args)
 
