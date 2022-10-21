# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from copy import deepcopy
import torch
import os
import shutil
import joblib 

def copy_file(source_path, target_path):
    shutil.copyfile(source_path, target_path)

def save_acc_predictor(args, acc_predictor):
    args.curr_acc_predictor_path = os.path.join(args.models_save_dir, f'acc_predictor_{args.curr_epoch}.joblib')
    with open(args.curr_acc_predictor_path, 'wb') as fp:
        joblib.dump(acc_predictor, fp)

def load_acc_predictor(args, predictor_saved_path=None):
    if predictor_saved_path is None:
        predictor_saved_path = args.curr_acc_predictor_path

    with open(predictor_saved_path, 'rb') as fp:
        acc_predictor = joblib.load(fp)
    return acc_predictor 


def load_search_checkpoints(args, model):
    resume_path = args.resume
    assert os.path.isfile(resume_path), "=> no checkpoint found at '{}'".format(resume_path)
    with open(resume_path, 'rb') as f:
        checkpoint = torch.load(f, map_location=torch.device('cpu'))

    print("=> loading checkpoint '{}'".format(resume_path))
    
    partial = checkpoint['state_dict']
    # print("ckpt:",partial)
    state = model.state_dict()
    pretrained_dict = {k: v for k, v in partial.items() if k in state and state[k].size() == partial[k].size()}
    state.update(pretrained_dict)
    model.load_state_dict(state)

    del checkpoint


def save_checkpoint(save_path, model, optimizer, lr_scheduler, args, epoch, best_acc_min, best_acc_max, best_epoch_min, best_epoch_max, is_best=False):
    save_state = {
        'epoch': epoch + 1,
        'args': args,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'best_acc_min': best_acc_min,
        'best_acc_max': best_acc_max,
        'best_epoch_min': best_epoch_min,
        'best_epoch_max': best_epoch_max,
    }

    best_model_path = os.path.join(
        os.path.dirname(save_path), 
        'best_{}'.format(os.path.basename(save_path))
    )

    with open(save_path, 'wb') as f:
        torch.save(save_state, f, _use_new_zipfile_serialization=False)

    if is_best: 
        copy_file(save_path, best_model_path)


def load_checkpoints(args, model, optimizer=None, lr_scheduler=None, logger=None):
    resume_path = args.resume
    assert os.path.isfile(resume_path), "=> no checkpoint found at '{}'".format(resume_path)
    with open(resume_path, 'rb') as f:
        checkpoint = torch.load(f, map_location=torch.device('cpu'))

    if logger:
        logger.info("=> loading checkpoint '{}'".format(resume_path))
    # args.start_epoch = checkpoint['epoch']
    best_acc_min = checkpoint['best_acc_min']
    best_acc_max = checkpoint['best_acc_max']
    # best_epoch = checkpoint['best_epoch']
    # TODO:
    # model.load_state_dict(checkpoint['state_dict'])
    partial = checkpoint['state_dict']
    # print("ckpt:",partial)
    state = model.state_dict()
    pretrained_dict = {k: v for k, v in partial.items() if k in state and state[k].size() == partial[k].size()}
    state.update(pretrained_dict)
    model.load_state_dict(state)

    # resume_with_a_different_optimizer = getattr(args, 'resume_with_a_different_optimizer', False)
    # resume_with_a_different_lr_scheduler = getattr(args, 'resume_with_a_different_lr_scheduler', False)
    # if optimizer and not resume_with_a_different_optimizer:
    #     optimizer.load_state_dict(checkpoint['optimizer'])

    # if lr_scheduler and not resume_with_a_different_optimizer and not resume_with_a_different_lr_scheduler:
    #     # use lr_scheduler settings defined in args
    #     skip_keys = list(args.lr_scheduler.__dict__.keys()) + ['clamp_lr']
    #     for k in skip_keys:
    #         if k in checkpoint['lr_scheduler']:
    #             checkpoint['lr_scheduler'].pop(k)
    #     lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    # elif lr_scheduler is not None:
    #     # reset lr_scheduler start epoch only
    #     lr_scheduler.step(checkpoint['lr_scheduler']['last_epoch'])

    if logger:
        logger.info("=> loaded checkpoint '{}' (epoch {})"
                    .format(resume_path, checkpoint['epoch']))

    del checkpoint
    
    return best_acc_min, best_acc_max
 

