#!/usr/bin/env python
# -*- coding:utf-8 -*-
# date: 2022/06
# author:Yushan Zheng
# emai:yszheng@buaa.edu.cn

"""
Modified from the training code for EfficientNet by 
Author: lukemelas (github username)
Github repo: https://github.com/lukemelas/EfficientNet-PyTorch
"""

from torch.utils.data import DistributedSampler
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data.distributed
import torch.utils.data
import torch.multiprocessing as mp
import torch.optim
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch
from yacs.config import CfgNode
import argparse
import builtins
import os

import math
import random
import shutil
import time
import warnings

import torch
import torchvision.transforms as transforms
from utils import *
from loader import CLPatchesDataset, DistributedWeightedSampler
from efficientnet_pytorch import EfficientNet
from byol import BYOL, GaussianBlur


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--cfg', type=str, default='',
                    help='The path of yaml config file')
parser.add_argument('--fold', type=int, default=0,
                    help='use all data for training if it is set -1')

parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
                         
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--image-size', default=224, type=int,
                    help='image size')
parser.add_argument('--multiprocessing-distributed', action='store_true', default=False,
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--eval-freq', type=int, default=5)

parser.add_argument('--label-id', type=int, default=1,
                    help='1 for all type classification, 2 for binary classification')
parser.add_argument('--od-input', action='store_true', default=False)
parser.add_argument('--weighted-sample', action='store_true')
parser.add_argument('--redo', action='store_true', default=False,
                    help='Ignore all the existing results and caches.')

# byol specific configs:
parser.add_argument('--hidden-dim', default=2048, type=int,
                    help='feature dimension (default: 2048)')
parser.add_argument('--pred-dim', default=256, type=int,
                    help='hidden dimension of the predictor (default: 256)')
parser.add_argument('--momentum-decay', default=0.99, type=float,
                    help='momentum of EMA updater (default: 0.99)')
parser.add_argument('--fix-pred-lr', action='store_true',
                    help='Fix learning rate for the predictor')

best_acc1 = 0

def main(args):
    if args.cfg:
        cfg = CfgNode(new_allowed=True)
        cfg.merge_from_file(args.cfg)
        merge_config_to_args(args, cfg)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.num_classes = args.task_list[args.label_id]['num_classes']

    checkpoint = []
    model_save_dir = get_contrastive_path(args)
    if not args.redo:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            print("=> loading checkpoint '{}'".format(args.resume))
        else:
            checkpoint_path = os.path.join(
                model_save_dir, 'checkpoint.pth.tar')
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(
                    checkpoint_path, map_location=torch.device('cpu'))
                print("=> loading checkpoint")

    if checkpoint:
        args.start_epoch = checkpoint['epoch']
        if args.start_epoch >= args.epochs:
            print('CNN training is finished')
            return 0
        else:
            print('CNN train from epoch {}/{}'.format(args.start_epoch, args.epochs))

    global best_acc1
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None and not args.distributed:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.rank == -1:
            if args.dist_url == "env://":
                args.rank = int(os.environ["RANK"])
            elif 'SLURM_PROCID' in os.environ:
                args.rank = int(os.environ['SLURM_PROCID'])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if 'efficientnet' in args.arch:  # NEW
        if args.pretrained:
            model = EfficientNet.from_pretrained(
                args.arch, num_classes=args.num_classes)
            print("=> using pre-trained model '{}'".format(args.arch))
        else:
            print("=> creating model '{}'".format(args.arch))
            model = EfficientNet.from_name(
                args.arch, num_classes=args.num_classes)
    else:
        if args.pretrained:
            print("=> using pre-trained model '{}'".format(args.arch))
            model = models.__dict__[args.arch](pretrained=True)
        else:
            print("=> creating model '{}'".format(args.arch))
            model = models.__dict__[args.arch](num_classes=args.num_classes)

    # BYOL
    learner = BYOL(model,args.hidden_dim, args.pred_dim, args.momentum_decay)
    init_lr = args.lr * args.batch_size / 256

    if args.distributed:
        learner = torch.nn.SyncBatchNorm.convert_sync_batchnorm(learner)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            learner.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            learner = torch.nn.parallel.DistributedDataParallel(
                learner, device_ids=[args.gpu])
        else:
            learner.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            learner = torch.nn.parallel.DistributedDataParallel(learner)

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        learner = learner.cuda(args.gpu)
        
    if args.fix_pred_lr:
            optim_params = [{'params': learner.module.online_encoder.parameters() if args.distributed
                                  else learner.online_encoder.parameters(), 'fix_lr': False},
                        {'params': learner.module.predictor.parameters() if args.distributed
                              else learner.predictor.parameters(), 'fix_lr': True}]
    else:
        optim_params = model.parameters()
    optimizer = torch.optim.SGD(optim_params, init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if checkpoint:
        learner.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> load state from checkpoint.")
    
    cudnn.benchmark = True

    traindir = os.path.join(get_data_list_path(args), args.fold_name, 'train')
    valdir = os.path.join(get_data_list_path(args), args.fold_name, 'val')
    
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    
    # Datasets
    train_set = CLPatchesDataset(traindir,
                             od_mode=args.od_input,
                             transform=train_transforms,
                             label_type=args.label_id
                             )
    val_set = CLPatchesDataset(valdir,
                             od_mode=args.od_input,
                             transform=train_transforms,
                             label_type=args.label_id
                             )
    if args.distributed:
        if args.weighted_sample:
            print('activate weighted sampling')
            train_sampler = DistributedWeightedSampler(
                train_set, train_set.get_weights(), args.world_size, args.rank)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    else:
        if args.weighted_sample:
            train_sampler = torch.utils.data.sampler.WeightedRandomSampler(
                train_set.get_weights(), len(train_set), replacement=True
            )
        else:
            train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None
    )
    
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank == 0):
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)                                            
        with open(model_save_dir + '.csv', 'a') as f:
            f.write('epoch, train loss, val loss\n')

    criterion = nn.CosineSimilarity(dim=1).cuda(args.gpu)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, init_lr, epoch, args)

        # train for one epoch
        train_loss = train(train_loader, learner, criterion, optimizer, epoch, args)
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank == 0):
            
            checkpoint_data = {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': learner.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }
            save_checkpoint(checkpoint_data, False,
                            os.path.join(model_save_dir, 'checkpoint.pth.tar'))

            save_single_module = {
                'arch': args.arch,
                'state_dict': learner.module.online_encoder.state_dict() if args.distributed else learner.online_encoder.state_dict(),
                'pretrained': args.pretrained,
                'level': args.level,
                'input_size': args.imsize,
                'intensity_thred': args.intensity_thred
            }
            save_checkpoint(
                save_single_module, True,
                os.path.join(model_save_dir, 'model_single_{}.pth.tar'.format(epoch)))

            if epoch % args.eval_freq == 0:
                val_loss = evaluate(val_loader, learner, criterion, epoch, args)

                with open(model_save_dir + '.csv', 'a') as f:
                    f.write('{},{:.2f},{:.2f}\n'.format(
                        epoch, train_loss, val_loss))

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses,
                 prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images_one, images_two) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpu is not None:
            images_one = images_one.cuda(args.gpu, non_blocking=True)
            images_two = images_two.cuda(args.gpu, non_blocking=True)

        # compute output
        p1, p2, z1, z2 = model(x1=images_one, x2=images_two)
        loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

        losses.update(loss.item(), images_one.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.print(i)

    return losses.avg


def evaluate(loader, model, criterion, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    progress = ProgressMeter(len(loader), batch_time, data_time, losses,
                 prefix="Epoch: [{}]".format(epoch))

    model.eval()

    end = time.time()
    with torch.no_grad():
         for i, (images_one, images_two) in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)
            if args.gpu is not None:
                images_one = images_one.cuda(args.gpu, non_blocking=True)
                images_two = images_two.cuda(args.gpu, non_blocking=True)

            # compute output
            p1, p2, z1, z2 = model(x1=images_one, x2=images_two)
            loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

            losses.update(loss.item(), images_one.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)

    return losses.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(
            os.path.dirname(filename), 'model_best.pth.tar'))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
