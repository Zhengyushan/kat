#!/usr/bin/env python
# -*- coding:utf-8 -*-
# date: 2022/06
# author:Yushan Zheng
# emai:yszheng@buaa.edu.cn

import sys

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
import os
import argparse
import time
import pickle
import cv2
import numpy as np
from sklearn.cluster import KMeans
from yacs.config import CfgNode

from byol.builder import replace_fc_with_mlp
from loader import SlideLocalTileDataset
from loader import get_tissue_mask, connectivity_and_dist
from efficientnet_pytorch import EfficientNet
from utils import *

parser = argparse.ArgumentParser('Extract cnn freatures of whole slide images')
parser.add_argument('--cfg', type=str, default='',
                    help='The path of yaml config file')    
parser.add_argument('--fold', type=int, default=0,
                    help='To identify the cnn used for feature extraction.\
                         a value -1 identify the cnn trained by all the training set data.\
                         It is useless when pretrained is set as True')
parser.add_argument('--resume', default='',
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--num-workers', type=int, default=0)
parser.add_argument('--dist-url', default='env://', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing-distributed', action='store_true', default=False,
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--invert-rgb', action='store_true', default=False,
                    help='Adjust the format between RGB and BGR\
                        The default color format of the patch is BGR')


def main(args):
    if args.cfg:
        cfg = CfgNode(new_allowed=True)
        cfg.merge_from_file(args.cfg)
        merge_config_to_args(args, cfg)

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    if args.distributed:
        if args.dist_url == "env://" and args.world_size == -1:
            args.world_size = int(os.environ["WORLD_SIZE"])
    ngpus_per_node = torch.cuda.device_count()

    args.wsi_feat_dir = get_graph_path(args)
    if not os.path.exists(args.wsi_feat_dir):
        os.makedirs(args.wsi_feat_dir)

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

    make_list(args)

def main_worker(gpu, ngpus_per_node, args):
    args.num_classes = args.task_list[args.label_id]['num_classes']

    args.gpu = gpu
    start_time = time.time()
    
    if args.gpu is not None:
        print("Use GPU: {} for encoding".format(args.gpu))

    if args.distributed:
        if args.rank == -1:
            if args.dist_url == "env://":
                args.rank = int(os.environ["RANK"])
            elif 'SLURM_PROCID' in os.environ:
                args.rank = int(os.environ['SLURM_PROCID'])
                
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + args.gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    
    
    # create model
    if 'efficientnet' in args.arch:
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
    
    if args.cl:
        model = replace_fc_with_mlp(model, args.hidden_dim, args.pred_dim)
        
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(
                args.resume, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['state_dict'])
    else:
        if args.cl:
            checkpoint = torch.load(os.path.join(get_contrastive_path(
                args), 'model_best.pth.tar'), map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(os.path.join(get_cnn_path(
                args), 'model_best.pth.tar'), map_location=torch.device('cpu'))
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
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    print('Load model time', time.time() - start_time)

    if args.pretrained:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    model.eval()
    if 'resnet' in args.arch:
        inference_model = nn.Sequential(*list(model.module.children())[:-1]) if args.distributed \
            else nn.Sequential(*list(model.children())[:-1])
    elif 'efficientnet' in args.arch:
        inference_model = model.module if args.distributed else model
    else:
        raise NotImplementedError('The network {} is not supported. \
            You may need to write the feature extraction code \
            for the network you choose.'.format(args.arch))
            
    with open(args.slide_list, 'rb') as f:
        slide_data = pickle.load(f)
    slide_list = slide_data['train'] + slide_data['test']
    
    current_slide_list = []
    for s_id, s_info in enumerate(slide_list):
        graph_save_path = os.path.join(args.wsi_feat_dir, '{}.pkl'.format(s_info[0]))
        if not os.path.exists(graph_save_path):
            current_slide_list.append(s_info)

    for s_id, s_info in enumerate(current_slide_list):
        porc_start = time.time()
        s_guid, s_rpath, s_label = s_info
        if args.distributed:
            # skip the slides the other gpus are working on
            if not s_id % args.world_size == args.rank:
                continue
        
        graph_save_path = os.path.join(args.wsi_feat_dir, '{}.pkl'.format(s_guid))
        if os.path.exists(graph_save_path):
            continue

        slide_path = os.path.join(args.slide_dir, s_rpath)
        image_dir = os.path.join(slide_path, scales[args.level])

        tissue_mask = get_tissue_mask(cv2.imread(
            os.path.join(slide_path, 'Overview.jpg')))
        content_mat = cv2.blur(
            tissue_mask, ksize=args.filter_size, anchor=(0, 0))
        content_mat = content_mat[::args.frstep, ::args.frstep] > args.intensity_thred
        
        patches_in_graph = np.sum(content_mat)
        if patches_in_graph < 1:
            continue

        # grid sampling
        sampling_mat = np.copy(content_mat)
        down_factor = 1
        if patches_in_graph > args.max_nodes:
            down_factor = int(np.sqrt(patches_in_graph/args.max_nodes)) + 1
            tmp = np.zeros(sampling_mat.shape, np.uint8) > 0
            tmp[::down_factor,::down_factor] = sampling_mat[::down_factor,::down_factor]
            sampling_mat = tmp
            patches_in_graph = np.sum(sampling_mat)

        # patch position
        patch_pos = np.transpose(np.asarray(np.where(sampling_mat)))
        # ajdacency_mat
        adj, re_dist = connectivity_and_dist(patch_pos, down_factor)
        
        # patch feature
        slide_dataset = SlideLocalTileDataset(image_dir, patch_pos*args.step, transform,
                                                args.tile_size, args.imsize, args.invert_rgb)
        slide_loader = torch.utils.data.DataLoader(
            slide_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True, drop_last=False)
        features = []
        for images in slide_loader:
            images = images.cuda(args.gpu, non_blocking=True)
            with torch.no_grad():
                if 'resnet' in args.arch:
                    x = inference_model(images)
                elif 'efficientnet' in args.arch:
                    x = inference_model._avg_pooling(inference_model.extract_features(images))
                else:
                    raise NotImplementedError('The network {} is not supported. \
                        You may need to write the feature extraction code \
                        for the network you choose.'.format(args.arch))

                features.append(x.cpu().numpy())
        features = np.concatenate(features, axis=0)

        npks = PATCH_NUMBER_PER_ANCHOR 
        if not args.npk in PATCH_NUMBER_PER_ANCHOR:
            npks.append(args.npk) 

        k_indexes, kns = [], []
        for npk in npks:
            kn = int(patches_in_graph/npk)+1 if int(patches_in_graph/npk) < np.sum(sampling_mat) else np.sum(sampling_mat) 
            kns.append(kn)
            kmeans_worker = KMeans(n_clusters=kn, random_state=9, max_iter=10)
            k_indexes.append(np.argmin(kmeans_worker.fit(patch_pos).transform(patch_pos), axis=0))

        # graph label
        graph_label = [args.task_list[task_id]['map'][s_label] \
            for task_id in args.task_list.keys()]

        with open(graph_save_path, 'wb') as f:
            graph = {
                'cm':content_mat,
                'feats':features,
                'adj':adj,
                'rd':re_dist,
                'knumber':kns,
                'npks': npks,
                'k_idx':k_indexes,
                'pos':patch_pos,
                'down_factor':down_factor,
                'label':graph_label
                }
            pickle.dump(graph, f)

        print('Processer #{}: {}/{} {}'.format(args.rank, s_id, len(current_slide_list), s_guid),
            '#patch:', patch_pos.shape[0], 
            '#kernel:', kns,
            'df:', down_factor, 
            'labels:', graph_label,
            'time:', time.time() - porc_start,
        )

def make_list(args):
    sample_list = []
    dataset_split_path = os.path.join(get_data_list_path(args), 'split.pkl')
    with open(dataset_split_path, 'rb') as f:
        folds = pickle.load(f)

    graph_list_dir = get_graph_list_path(args)
    for f_id, fold_list in enumerate(folds):
        sub_set_name = 'test' if f_id==args.fold_num else 'fold_{}'.format(f_id)

        sample_list_fold = []
        for s_id, s_guid in enumerate(fold_list):
            if isinstance(s_guid, list):
                s_guid = s_guid[0]
            slide_graph_path = os.path.join(args.wsi_feat_dir, s_guid)
            if os.path.exists(slide_graph_path + '.pkl'):         
                with open(slide_graph_path + '.pkl', 'rb') as f:
                    graph = pickle.load(f)
                sample_list_fold.append((s_guid + '.pkl', graph['label'], s_id))

        sample_list.append(sample_list_fold)

    for f_id in range(args.fold_num+1):
        sub_set_name = 'list_fold_all' if f_id==args.fold_num else 'list_fold_{}'.format(f_id)
        val_set = sample_list[f_id]

        train_set = []
        if f_id == args.fold_num:
            for train_f_id in range(args.fold_num+1):
                train_set += sample_list[train_f_id]
        else:
            train_index = np.hstack((np.arange(0, f_id),np.arange(f_id+1,args.fold_num)))
            for train_f_id in train_index:
                train_set += sample_list[train_f_id]

        train_set_shuffle = []
        for tss in np.random.permutation(len(train_set)):
            train_set_shuffle.append(train_set[tss])
        test_set = sample_list[-1]

        sub_list_path = os.path.join(graph_list_dir, sub_set_name)
        if not os.path.exists(sub_list_path):
            os.makedirs(sub_list_path)

        with open(os.path.join(sub_list_path,'train'), 'wb') as f:
            pickle.dump({
                'base_dir':args.wsi_feat_dir, 
                'list':train_set, 
                }, f)

        if len(val_set):
            with open(os.path.join(sub_list_path,'val'), 'wb') as f:
                pickle.dump({
                    'base_dir':args.wsi_feat_dir, 
                    'list':val_set, 
                    }, f)

        if len(test_set):
            with open(os.path.join(sub_list_path,'test'), 'wb') as f:
                pickle.dump({
                    'base_dir':args.wsi_feat_dir, 
                    'list':test_set, 
                    }, f)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
