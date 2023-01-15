#!/usr/bin/env python
# -*- coding:utf-8 -*-
# date: 2022/06
# author:Yushan Zheng
# emai:yszheng@buaa.edu.cn

import os
import pickle
import math
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from tabulate import tabulate
import numpy as np
import torch

# The definition of magnification of our gastic dataset.
# 'Large':40X, 'Medium':20X, 'Small':10X, 'Overview':5X
scales = ['Large', 'Medium', 'Small', 'Overview']

# The default number of patches for a kernel
PATCH_NUMBER_PER_ANCHOR = [36, 64, 100, 144, 256, 400]

def merge_config_to_args(args, cfg):
    # dirs
    args.data_conf_dir = cfg.DATA.DATASET_CONFIG_DIR
    args.slide_dir = cfg.DATA.LOCAL_SLIDE_DIR
    args.patch_dir = os.path.join(cfg.DATA.DATA_SAVE_DIR, 'patch')
    args.list_dir = os.path.join(cfg.DATA.DATA_SAVE_DIR, 'patch_list')
    args.cnn_dir = os.path.join(cfg.DATA.DATA_SAVE_DIR, 'cnn_model')
    args.contrst_dir = os.path.join(cfg.DATA.DATA_SAVE_DIR, 'contrastive_model')
    args.feat_dir = os.path.join(cfg.DATA.DATA_SAVE_DIR, 'cnn_feat')
    args.graph_dir = os.path.join(cfg.DATA.DATA_SAVE_DIR, 'graph')
    args.graph_list_dir = os.path.join(cfg.DATA.DATA_SAVE_DIR, 'graph_list')
    args.kat_dir = os.path.join(cfg.DATA.DATA_SAVE_DIR, 'kat_model')
    

    # data
    args.slide_list = os.path.join(args.data_conf_dir, 'slide_list.pkl')
    args.task_list, args.lesions = get_slide_config(os.path.join(args.data_conf_dir, 'dataset_config.pkl'))
    args.label_id = cfg.DATA.LABEL_ID
    args.test_ratio = cfg.DATA.TEST_RATIO
    args.fold_num = cfg.DATA.FOLD_NUM

    # image
    if 'IMAGE' in cfg:
        args.level = cfg.IMAGE.LEVEL
        args.mask_level = cfg.IMAGE.MASK_LEVEL
        args.imsize = cfg.IMAGE.PATCH_SIZE
        args.tile_size = cfg.IMAGE.LOCAL_TILE_SIZE
        args.rl = args.mask_level-args.level
        args.msize = args.imsize >> args.rl
        args.mhalfsize = args.msize >> 1

    # sampling
    if 'SAMPLE' in cfg:
        args.positive_ratio = cfg.SAMPLE.POS_RAT
        args.negative_ratio = cfg.SAMPLE.NEG_RAT
        args.intensity_thred = cfg.SAMPLE.INTENSITY_THRED
        args.sample_step = cfg.SAMPLE.STEP
        args.max_per_class = cfg.SAMPLE.MAX_PER_CLASS
        args.save_mask = cfg.SAMPLE.SAVE_MASK
        
        args.srstep = args.sample_step>>args.rl
        args.filter_size = (args.imsize>>args.rl, args.imsize>>args.rl)
        
    # CNN
    if 'CNN' in cfg:
        args.arch = cfg.CNN.ARCH
        args.pretrained = cfg.CNN.PRETRAINED
        args.cl = cfg.CNN.CONTRASTIVE
        args.freeze_feat = cfg.CNN.FREEZE_FEAT
        if args.cl:
            args.hidden_dim = cfg.CNN.BYOL.HIDDEN_DIM
            args.pred_dim = cfg.CNN.BYOL.PRE_DIM
            args.momentum_decay = cfg.CNN.BYOL.M_DECAY
            args.fix_pred_lr = cfg.CNN.BYOL.FIX_PRED_LR

    # WSI feature
    if 'FEATURE' in cfg:
        args.step = cfg.FEATURE.STEP
        args.frstep = args.step>>args.rl
        args.max_nodes = cfg.FEATURE.MAX_NODES

    if 'VIT' in cfg:
        args.trfm_depth = args.trfm_depth if ('trfm_depth' in args and args.trfm_depth) else cfg.VIT.DEPTH
        args.trfm_heads = args.trfm_heads if ('trfm_heads' in args and args.trfm_heads) else cfg.VIT.HEADS
        args.trfm_dim = cfg.VIT.DIM
        args.trfm_mlp_dim = cfg.VIT.MLP_DIM
        args.trfm_dim_head = cfg.VIT.HEAD_DIM
        args.trfm_pool = cfg.VIT.POOL

    if 'KAT' in cfg:
        args.npk = args.npk if ('npk' in args and args.npk) else cfg.KAT.PATCH_PER_KERNEL
        args.kn = int(args.max_nodes/args.npk) + 1
        
        args.p_dim = args.p_dim if ('p_dim' in args and args.p_dim) else cfg.KAT.BYOL.PROJECTOR_DIM
        args.aug_rate = args.aug_rate if ('aug_rate' in args and args.aug_rate) else cfg.KAT.BYOL.NODE_AUG
        args.sl_weight = args.sl_weight if ('sl_weight' in args and args.sl_weight) else cfg.KAT.BYOL.SL_WEIGHT

    return args


def get_sampling_path(args):
    prefix = '[l{}t{}s{}m{}][p{}n{}i{}]'.format(args.level, args.imsize,
                                              args.sample_step, args.max_per_class,
                                              int(args.positive_ratio * 100),
                                              int(args.negative_ratio * 100),
                                              args.intensity_thred)

    return os.path.join(args.patch_dir, prefix)

def get_data_list_path(args):
    prefix = get_sampling_path(args)
    prefix = '{}[f{}_t{}]'.format(prefix[prefix.find('['):], args.fold_num,
                                int(args.test_ratio * 100))

    return os.path.join(args.list_dir, prefix)

def get_cnn_path(args):
    prefix = get_data_list_path(args)
    args.fold_name = 'list_fold_all' if args.fold == -1 else 'list_fold_{}'.format(
        args.fold)
    prefix = '{}[{}_td_{}_{}]'.format(prefix[prefix.find('['):], args.arch, 
                args.label_id, args.fold_name)
    if args.freeze_feat:
        prefix += '[frz]'
    return os.path.join(args.cnn_dir, prefix)

def get_contrastive_path(args):
    prefix = get_data_list_path(args)
    args.fold_name = 'list_fold_all' if args.fold == -1 else 'list_fold_{}'.format(
        args.fold)
    prefix = '{}[{}_td_{}_{}]'.format(prefix[prefix.find('['):], args.arch, args.label_id if not args.cl else 'cl',
                                    args.fold_name)

    return os.path.join(args.contrst_dir, prefix)

def get_feature_path(args):
    if args.pretrained:
        prefix = '[{}_pre][fs{}]'.format(args.arch, args.step)
    else:
        prefix = get_data_list_path(args)
        args.fold_name = 'list_fold_all' if args.fold == -1 else 'list_fold_{}'.format(
                        args.fold)
        prefix = '{}[{}_td_{}][fs{}][{}]'.format(prefix[prefix.find('['):], 
            args.arch, args.label_id if not args.cl else 'cl', args.step, args.fold_name)

        if args.freeze_feat:
            prefix += '[frz]'

    return os.path.join(args.feat_dir, prefix)


def get_graph_path(args):
    prefix = get_feature_path(args)
    prefix = '{}[m{}]'.format(prefix[prefix.find('['):], 
        args.max_nodes)

    return os.path.join(args.graph_dir, prefix)

def get_graph_list_path(args):
    prefix = get_feature_path(args)
    prefix = '{}[m{}]'.format(prefix[prefix.find('['):], 
        args.max_nodes)

    return os.path.join(args.graph_list_dir,prefix)


def get_slide_config(config_path):
    with open(config_path, 'rb') as f:
        data = pickle.load(f)

    return data['tasks'], data['lesions']


def get_kat_path(args, prefix_name=''):
    prefix = get_graph_list_path(args)
    prefix = '{}[d{}_h_{}_de{}dm{}dh{}_{}][npk_{}][t{}]'.format(prefix_name+prefix[prefix.find('['):], 
        args.trfm_depth, args.trfm_heads, args.trfm_dim, args.trfm_mlp_dim, args.trfm_dim_head, args.trfm_pool,
        args.npk,
        args.label_id
        )

    return os.path.join(args.kat_dir, prefix)
    
def get_kat_byol_path(args, prefix_name=''):
    prefix = get_graph_list_path(args)
    prefix = '{}[d{}_h_{}_de{}dm{}dh{}_{}][npk_{}][ar_{}_pd_{}_slw{}][t{}]'.format(prefix_name+prefix[prefix.find('['):], 
        args.trfm_depth, args.trfm_heads, args.trfm_dim, args.trfm_mlp_dim, args.trfm_dim_head, args.trfm_pool,
        args.npk, args.aug_rate, args.p_dim, args.sl_weight,
        args.label_id
        )

    return os.path.join(args.kat_dir, prefix)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr

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

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]