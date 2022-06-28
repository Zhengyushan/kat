#!/usr/bin/env python
# -*- coding:utf-8 -*-
# date: 2020/12
# author:yushan zheng
# emai:yszheng@buaa.edu.cn

import argparse
import os
import pickle
import time
import numpy as np
from yacs.config import CfgNode

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.distributed
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist

from tabulate import tabulate
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from model import KAT, kat_inference
from loader import KernelWSILoader
from loader import DistributedWeightedSampler
from utils import *

import random
import builtins
import warnings

def arg_parse():
    parser = argparse.ArgumentParser(description='GCN-Hash arguments.')

    parser.add_argument('--cfg', type=str,
            default='',
            help='The path of yaml config file')

    parser.add_argument('--fold', type=int, default=-1, help='use all data for training if it is set -1')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size.')
    parser.add_argument('--num-epochs', type=int, default=300,
                        help='Number of epochs to train.')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of workers to load data.')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate.')
    parser.add_argument('--shuffle-train', default=False, action='store_true',
                        help='Shuffle the train list')
    parser.add_argument('--weighted-sample', action='store_true',
                        help='Balance the sample number from different types\
                              in each mini-batch for training.')

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
    parser.add_argument('--multiprocessing-distributed', action='store_true', default=False,
                        help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    
    parser.add_argument('--redo', default=False, action='store_true',
                        help='Ingore all the cache files and re-train the model.')
    parser.add_argument('--eval-model', type=str, default='',
                        help='provide a path of a trained model to evaluate the performance')
    parser.add_argument('--eval-freq', type=int, default=30,
                        help='The epoch frequency to evaluate on vlidation and test sets.')
    parser.add_argument('--print-freq', type=int, default=10,
                        help='The mini-batch frequency to print results.')
    parser.add_argument('--prefix-name', type=str, default='',
                        help='A prefix for the model name.')
    
    parser.add_argument('--node-aug', default=False, action='store_true',
                        help='Randomly reduce the nodes for data augmentation》')

    return parser.parse_args()


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
    graph_model_path = get_kat_path(args, args.prefix_name)

    checkpoint = []
    if not args.redo:
        checkpoint_path = os.path.join(
            graph_model_path, 'checkpoint.pth.tar')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(
                checkpoint_path, map_location=torch.device('cpu'))
            print("=> loading checkpoint")

    if checkpoint:
        args.start_epoch = checkpoint['epoch']
        if args.start_epoch >= args.num_epochs:
            print('model training is finished')
            return 0
        else:
            print('model train from epoch {}/{}'.format(args.start_epoch, args.num_epochs))
    else:
        args.start_epoch = 0

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

    graph_list_dir = os.path.join(get_graph_list_path(args), args.fold_name)
    # train graph data
    train_set = KernelWSILoader(
            os.path.join(graph_list_dir, 'train'),
            max_node_number=args.max_nodes,
            patch_per_kernel=args.npk,
            task_id=args.label_id,
            max_kernel_num=args.kn,
            node_aug=args.node_aug,
            two_augments=False
            )

    args.input_dim = train_set.get_feat_dim()
    # create model
    model = KAT(
        num_pk=args.npk,
        patch_dim=args.input_dim,
        num_classes=args.num_classes, 
        dim=args.trfm_dim, 
        depth=args.trfm_depth, 
        heads=args.trfm_heads, 
        mlp_dim=args.trfm_mlp_dim, 
        dim_head=args.trfm_dim_head, 
        num_kernal=args.kn,
        pool = args.trfm_pool, 
    )

    if args.gpu is not None:
        model = model.cuda(args.gpu)

    if os.path.isfile(args.resume):
        print("=> resume checkpoint '{}'".format(args.resume))
        resume_model_params = torch.load(
            args.resume, map_location=torch.device('cpu'))
        model.load_state_dict(resume_model_params['state_dict'])
    else:
        if checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.num_workers = int(args.num_workers / ngpus_per_node)

            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    if args.weighted_sample:
        print('activate weighted sampling')
        if args.distributed:
            train_sampler = DistributedWeightedSampler(
                train_set, train_set.get_weights(), args.world_size, args.rank)
        else:
            train_sampler = torch.utils.data.sampler.WeightedRandomSampler(
                train_set.get_weights(), len(train_set), replacement=True
            )
    else:
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_set)
        else:
            train_sampler = None

    train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=args.shuffle_train,
            num_workers=args.num_workers, sampler=train_sampler)

    # validation graph data
    val_path = os.path.join(graph_list_dir, 'val')
    if not os.path.exists(val_path):
        valid_loader = None
    else:
        valid_set = KernelWSILoader(val_path,
            max_node_number=args.max_nodes,
            patch_per_kernel=args.npk,
            task_id=args.label_id,
            max_kernel_num=args.kn,
            node_aug=False,
            two_augments=False
            )
        valid_loader = torch.utils.data.DataLoader(
            valid_set, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, drop_last=False, sampler=None
            )
        
    # test graph data
    test_path = os.path.join(graph_list_dir, 'test')
    if not os.path.exists(test_path):
        test_loader = None
    else:
        test_set = KernelWSILoader(test_path,
            max_node_number=args.max_nodes,
            patch_per_kernel=args.npk,
            task_id=args.label_id,
            max_kernel_num=args.kn,
            node_aug=False,
            two_augments=False
            )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, drop_last=False, sampler=None
            )

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.7)

    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    if args.eval_model and test_loader is not None:
        model_params = torch.load(args.eval_model, map_location='cpu')
        model.load_state_dict(model_params['state_dict'])

        test_acc, test_cm, test_auc, test_data = evaluate(test_loader, model, criterion, args, 'Valid')

        with open(os.path.join(graph_model_path,  'eval.pkl'), 'wb') as f:
            pickle.dump({'acc':test_acc, 'cm':test_cm, 'auc':test_auc,'data':test_data}, f)

        return 0

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank == 0):    
        if not os.path.exists(graph_model_path):
            os.makedirs(graph_model_path)
        with open(graph_model_path + '.csv', 'a') as f:
            f.write('epoch, train acc, V, val acc, val w-auc, val m-auc, val w-f1, val m-f1 ,\
                T, tet acc, test w-auc, test m-auc, test w-f1, test m-f1, \n')

    for epoch in range(args.start_epoch, args.num_epochs):
        begin_time = time.time()

        train_acc = train(train_loader, model, criterion, optimizer, epoch, args)
        print('epoch time: ', time.time()-begin_time)
        scheduler.step()
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank == 0):
            if epoch % args.eval_freq == 0:
                if valid_loader is not None:
                    val_acc, val_cm, val_auc, val_data = evaluate(valid_loader, model, criterion, args, 'Valid')

                if test_loader is not None:
                    test_acc, test_cm, test_auc, test_data = evaluate(test_loader, model, criterion, args, 'Test')

                if valid_loader is not None:
                    with open(graph_model_path + '.csv', 'a') as f:
                        f.write('{},{:.3f},V,{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},T,{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}, SUB,'.format(
                            epoch, train_acc/100.0, 
                                val_acc/100.0, val_auc['micro'], val_auc['macro'], val_auc['w_f1'], val_auc['m_f1'],
                                test_acc/100.0, test_auc['micro'], test_auc['macro'], test_auc['w_f1'], test_auc['m_f1'],)
                                )
                        for cn in range(test_cm.shape[0]):
                            f.write(',{:.2f}'.format(test_cm[cn, cn]))
                        f.write('\n') 

                result_data_path = os.path.join(graph_model_path, 'result{}.pkl'.format(epoch + 1))
                with open(result_data_path, 'wb') as f:
                    pickle.dump({'val':val_data, 'test':test_data}, f)

                torch.save({
                        'epoch': epoch + 1,
                        'state_dict': model.module.state_dict() if args.distributed else model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                    }, os.path.join(graph_model_path, 'checkpoint.pth.tar'))

                torch.save({
                        'epoch': epoch + 1,
                        'state_dict': model.module.state_dict() if args.distributed else model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'args': args
                    }, os.path.join(graph_model_path, 'model_{}.pth.tar'.format(epoch + 1)))


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top2 = AverageMeter('Acc@2', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                             top2, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (data, label) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = label.cuda(non_blocking=True)

        # compute output
        _, output = kat_inference(model, data)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc2 = accuracy(F.softmax(output, dim=1), target, topk=(1, 2))
        losses.update(loss.item(), target.size(0))
        top1.update(acc1[0], target.size(0))
        top2.update(acc2[0], target.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.print(i)

    return top1.avg


def evaluate(val_loader, model, criterion, args, prefix='Test'):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top2 = AverageMeter('Acc@2', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top2,
                             prefix=prefix)

    # switch to evaluate mode
    model.eval()
    y_preds = []
    y_labels = []
    end = time.time()
    
    processing_time = 0
    with torch.no_grad():
        for i, (data, label) in enumerate(val_loader):
            target = label.cuda(non_blocking=True)
            # compute output
            pro_start = time.time()
            _, output = kat_inference(model, data)
            processing_time += (time.time() - pro_start)
            loss = criterion(output, target)

            y_preds.append(F.softmax(output, dim=1).cpu().data)
            y_labels.append(label)
            # measure accuracy and record loss
            acc1, acc2 = accuracy(output, target, topk=(1, 2))
            losses.update(loss.item(), target.size(0))
            top1.update(acc1[0], target.size(0))
            top2.update(acc2[0], target.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)
        
        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@2 {top2.avg:.3f} Sample per Second {time:.3f}'
              .format(top1=top1, top2=top2, time=len(val_loader)*args.batch_size/processing_time))

    y_preds = torch.cat(y_preds)
    y_labels = torch.cat(y_labels)
    confuse_mat, auc = calc_classification_metrics(y_preds, y_labels, args.num_classes, prefix=prefix)

    return top1.avg, confuse_mat, auc, {'pred':y_preds, 'label':y_labels}


def calc_classification_metrics(y_preds, y_labels, num_classes=None, prefix='Eval'):
    if num_classes is None:
        num_classes = max(y_labels) + 1

    y_labels = y_labels.numpy()
    y_preds = y_preds.numpy()

    results = {}

    results["m_f1"] = f1_score(y_labels, np.argmax(y_preds, axis=1), average='macro')
    results["w_f1"] = f1_score(y_labels, np.argmax(y_preds, axis=1), average='weighted')
    if num_classes < 3:
        results["macro"] = roc_auc_score(y_labels, y_preds[:,1], average='macro', multi_class='ovo')
        results["micro"] = roc_auc_score(y_labels, y_preds[:,1], average='weighted', multi_class='ovr')
    else:
        results["macro"] = roc_auc_score(y_labels, y_preds, average='macro', multi_class='ovo')
        results["micro"] = roc_auc_score(y_labels, y_preds, average='weighted', multi_class='ovr')

    confuse_mat = confusion_matrix(
        y_labels, np.argmax(y_preds, axis=1))
    confuse_mat = np.asarray(confuse_mat, float)

    values = [prefix, results['micro'], results['macro'], results['w_f1'], results['m_f1']]
    headers = ['Classification', 'weighted auc', 'macro auc', 'weighted f1', 'macro f1']
    for y in range(max(y_labels)+1):
        confuse_mat[y, :] = confuse_mat[y, :]/np.sum(y_labels == y)
        values.append(confuse_mat[y, y])
        headers.append(str(y))

    print(tabulate([values,], headers, tablefmt="grid"))

    return confuse_mat, results


def accuracy(output, target, topk=(1,2)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    

if __name__ == "__main__":
    args = arg_parse()
    main(args)
