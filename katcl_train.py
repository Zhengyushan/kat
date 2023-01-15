#!/usr/bin/env python
# -*- coding:utf-8 -*-
# date: 2020/12
# author:yushan zheng
# emai:yszheng@buaa.edu.cn

import argparse
import os
import shutil
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


from model import KATCL, kat_inference
from loader import KernelWSILoader
from utils import *
from loader import DistributedWeightedSampler

import random
import builtins
import warnings

def arg_parse():
    parser = argparse.ArgumentParser(description='KAT arguments.')

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
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate.')
    parser.add_argument("--warmup-epochs", default=50, type=int,
                        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min-lr', type=float, default=1e-6, help="""Target LR at the
                            end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--weight-decay', type=float, default=0.04, help="""Initial value of the
                            weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight-decay-end', type=float, default=0.4, help="""Final value of the
                            weight decay. We use a cosine schedule for WD and using a larger decay by
                            the end of training improves performance for ViTs.""")

    parser.add_argument('--shuffle-train', default=False, action='store_true',
                        help='Shuffle the train list')
    parser.add_argument('--weighted-sample', action='store_true', default=False,
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
    parser.add_argument('--eval-model', type=str, default=None,
                        help='provide a path of a trained model to evaluate the performance')
    parser.add_argument('--eval-freq', type=int, default=30,
                        help='The epoch frequency to evaluate on vlidation and test sets.')
    parser.add_argument('--print-freq', type=int, default=10,
                        help='The mini-batch frequency to print results.')
    parser.add_argument('--prefix-name', type=str, default='',
                        help='A prefix for the model name.')
    
    parser.add_argument('--p-dim', type=int, default=None,
                        help='Projector output dimension')

    parser.add_argument('--aug-rate', type=float, default=None,)
    parser.add_argument('--sl-weight', type=float, default=0.0,)

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
    graph_model_path = get_kat_byol_path(args, args.prefix_name)

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
        if args.start_epoch >= args.num_epochs and not args.eval_model:
            print('model training is finished')
            return 0
        else:
            print('model train from epoch {}/{}'.format(args.start_epoch, args.num_epochs))
    else:
        args.start_epoch = 0

    args.gpu = gpu
    print('Using gpu', args.gpu)
    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None and not args.distributed:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
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
            aug_rate=args.aug_rate,
            two_augments=True if not args.eval_model else False
            )
    args.input_dim = train_set.get_feat_dim()


    model = KATCL(
        num_pk=args.npk,
        patch_dim=args.input_dim,
        num_classes=args.num_classes, 
        dim=args.trfm_dim, 
        depth=args.trfm_depth, 
        heads=args.trfm_heads, 
        mlp_dim=args.trfm_mlp_dim, 
        dim_head=args.trfm_dim_head, 
        num_kernal=args.kn,
        pool=args.trfm_pool, 
        byol_hidden_dim=args.p_dim * 4,
        byol_pred_dim=args.p_dim
    )

    if args.gpu is not None:
        model = model.cuda(args.gpu)
    if args.eval_model is not None:
        print("=> eval model '{}'".format(args.eval_model))
        model_params = torch.load(args.eval_model, map_location='cpu')
        model.load_state_dict(model_params['state_dict'])
    else:
        if os.path.isfile(args.resume):
            print("=> resume from model '{}'".format(args.resume))
            resume_model_params = torch.load(
                args.resume, map_location=torch.device('cpu'))
            model.load_state_dict(resume_model_params['state_dict'])
        else:
            if checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
                print("=> resume checkpoint '{}'".format(args.start_epoch))
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
        if args.distributed and not args.eval_model:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_set)
        else:
            train_sampler = None

    train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=args.shuffle_train,
            num_workers=args.num_workers, sampler=train_sampler, pin_memory=True)

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
            node_aug=False
            )
        valid_loader = torch.utils.data.DataLoader(
            valid_set, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, drop_last=False, sampler=None, pin_memory=True
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
            node_aug=False
            )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, drop_last=False, sampler=None, pin_memory=True
            )

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    sl_criterion = nn.CosineSimilarity(dim=-1).cuda(args.gpu)
    params_groups = get_params_groups(model)
    optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    lr_schedule = cosine_scheduler(
        args.lr * (args.batch_size * max(args.world_size, 1)) / 32.,  # linear scaling rule
        args.min_lr,
        args.num_epochs, len(train_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.num_epochs, len(train_loader),
    )

    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    if args.eval_model:
        eval_model = model.module if args.distributed else model
        for cur_loader, cur_prefix in zip((train_loader, valid_loader, test_loader), ('train', 'val', 'test')):
            if cur_loader is not None:
                test_acc, test_cm, test_auc, test_data = evaluate(cur_loader, eval_model, criterion, args, cur_prefix)
                with open(os.path.join(graph_model_path, cur_prefix + '_eval.pkl'), 'wb') as f:
                    pickle.dump({'acc':test_acc, 'cm':test_cm, 'auc':test_auc,'data':test_data}, f)

        return 0
        
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank == 0):
        if not os.path.exists(graph_model_path):
            os.makedirs(graph_model_path)
        
        with open(graph_model_path + '.csv', 'a') as f:
            f.write('epoch, train acc, V, val acc, val w-auc, val m-auc, val w-f1, val m-f1,\
             T, tet acc, test w-auc, test m-auc, test w-f1, test m-f1, \n')

    for epoch in range(args.start_epoch, args.num_epochs):
        begin_time = time.time()

        train_acc = train(train_loader, model, criterion, sl_criterion, optimizer, lr_schedule, wd_schedule, epoch, args)
        print('epoch time: ', time.time()-begin_time)
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank == 0):
            if epoch % args.eval_freq == 0:
                with open(graph_model_path + '.csv', 'a') as f:
                    f.write('{},{:.3f},'.format(epoch, train_acc/100.0,))
                eval_model = model.module if args.distributed else model
                if valid_loader is not None:
                    val_acc, val_cm, val_auc, val_data = evaluate(valid_loader, eval_model, criterion, args, 'Valid')
                    with open(graph_model_path + '.csv', 'a') as f:
                        f.write('V,{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},'.format(
                                val_acc/100.0, val_auc['micro'], val_auc['macro'], val_auc['w_f1'], val_auc['m_f1'],)
                                )
                    result_data_path = os.path.join(graph_model_path, 'val_result{}.pkl'.format(epoch + 1))
                    with open(result_data_path, 'wb') as f:
                        pickle.dump(val_data, f)

                if test_loader is not None:
                    test_acc, test_cm, test_auc, test_data = evaluate(test_loader, eval_model, criterion, args, 'Test')
                    with open(graph_model_path + '.csv', 'a') as f:
                        f.write('T,{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}, SUB,'.format(
                                test_acc/100.0, test_auc['micro'], test_auc['macro'], test_auc['w_f1'], test_auc['m_f1'],)
                                )
                        for cn in range(test_cm.shape[0]):
                            f.write(',{:.2f}'.format(test_cm[cn, cn]))
                        f.write('\n') 

                    result_data_path = os.path.join(graph_model_path, 'test_result{}.pkl'.format(epoch + 1))
                    with open(result_data_path, 'wb') as f:
                        pickle.dump(test_data, f)

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

def train(train_loader, model, criterion, sl_criterion, optimizer, lr_schedule, wd_schedule, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses1 = AverageMeter('Loss1', ':.4e')
    losses2 = AverageMeter('Loss2', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top2 = AverageMeter('Acc@2', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses1, losses2, top1,
                             top2, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (data1, data2, label) in enumerate(train_loader):
        it = len(train_loader) * epoch + i  # global training iteration
        for g_idx, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if g_idx == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        p1, o1, z2 = model(data1, data2)
        target = label.cuda(non_blocking=True)
        cls_loss = criterion(o1, target)
        sl_loss = - sl_criterion(p1, z2).mean() + 1
        loss = cls_loss + args.sl_weight*sl_loss if args.sl_weight>1e-7 else cls_loss

        # measure accuracy and record loss
        acc1, acc2 = accuracy(F.softmax(o1, dim=1), target, topk=(1, 2))
        losses1.update(cls_loss.item(), label.size(0))
        losses2.update(sl_loss.item(), label.size(0))
        top1.update(acc1[0], label.size(0))
        top2.update(acc2[0], label.size(0))

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
    y_logits = []
    y_labels = []
    end = time.time()

    with torch.no_grad():
        for i, (data, label) in enumerate(val_loader):
            target = label.cuda(non_blocking=True)
            # compute output
            _, output = kat_inference(model.online_kat, data)
            loss = criterion(output, target)

            y_preds.append(F.softmax(output, dim=1).cpu().data)
            y_logits.append(output.cpu().data)
            y_labels.append(label)

            # measure accuracy and record loss
            acc1, acc2 = accuracy(F.softmax(output, dim=1), target, topk=(1, 2))
            losses.update(loss.item(), target.size(0))
            top1.update(acc1[0], target.size(0))
            top2.update(acc2[0], target.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)

    y_preds = torch.cat(y_preds)
    y_logits = torch.cat(y_logits)
    y_labels = torch.cat(y_labels)
    confuse_mat, auc = calc_classification_metrics(y_preds, y_labels, args.num_classes, prefix=prefix)

    return top1.avg, confuse_mat, auc, {'pred':y_logits, 'label':y_labels}


if __name__ == "__main__":
    args = arg_parse()
    main(args)
