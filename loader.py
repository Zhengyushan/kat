#!/usr/bin/env python
# -*- coding:utf-8 -*-
# date: 2022/06
# author:Yushan Zheng
# emai:yszheng@buaa.edu.cn

import os
import pickle
import cv2
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data


class CLPatchesDataset(data.Dataset):
    def __init__(self, file_path, transform, od_mode=True, label_type=1):
        self.transform = transform
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        self.data_dir = data['base_dir']
        self.image_list = data['list']
        self.od = od_mode
        self.lt = label_type

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.image_list[index][0])).convert('RGB')

        if self.transform!=None:
            img1 = self.transform(img)
            img2 = self.transform(img)
        if self.od:
            img1 = -torch.log(img + 1.0/255.0)
            img2 = -torch.log(img + 1.0/255.0)

        return img1, img2

    def __len__(self):
        return len(self.image_list)

    def get_weights(self):
        num = self.__len__()
        labels = np.zeros((num,), np.int)
        for s_ind, s in enumerate(self.image_list):
            labels[s_ind] = s[self.lt]
        tmp = np.bincount(labels)
        weights = 1.0 / np.asarray(tmp[labels], np.float)

        return weights


class SlideLocalTileDataset(data.Dataset):
    def __init__(self, image_dir, position_list, transform,
            tile_size=512, imsize=224, od_mode=False, invert_rgb=False):
        self.transform = transform

        self.im_dir = image_dir
        self.pos = position_list
        self.od = od_mode
        self.ts = tile_size
        self.imsize = imsize
        self.inv_rgb = invert_rgb

    def __getitem__(self, index):
        img = extract_tile(self.im_dir, self.ts, self.pos[index][1], self.pos[index][0], self.imsize, self.imsize)
        if len(img) == 0:
            img = np.ones((self.imsize, self.imsize, 3), np.uint8) * 240
        if self.inv_rgb:
            img = img[:,:,[2,1,0]]
        img = Image.fromarray(img).convert('RGB')
        img = self.transform(img)

        if self.od:
            img = -torch.log(img + 1.0/255.0)

        return img

    def __len__(self):
        return self.pos.shape[0]


class KernelWSILoader(torch.utils.data.Dataset):
    def __init__(self, list_path, max_node_number, patch_per_kernel, task_id=1,
                max_kernel_num=16, node_aug=True, aug_rate=0.5, two_augments=False, 
                ):
        with open(list_path, 'rb') as f:
            data = pickle.load(f)
        self.dl = data['list']
        self.list_dir = data['base_dir']

        self.maxno = max_node_number
        self.ti=task_id
        self.ar = aug_rate

        with open(self.get_wsi_data_path(0), 'rb') as f:
            wsi_data = pickle.load(f)

        self.feat_dim = wsi_data['feats'].shape[1]
        self.nk = max_kernel_num

        if patch_per_kernel not in wsi_data['npks']:
            raise NotImplementedError('Do not support nk = {}.'.format(self.nk))
        self.nk_lvl = np.where(np.asarray(wsi_data['npks'])==patch_per_kernel)[0][0]

        self.na = node_aug
        self.two = two_augments
        
    def __len__(self):
        return len(self.dl)

    def __getitem__(self, idx):
        with open(self.get_wsi_data_path(idx), 'rb') as f:
            wsi_data = pickle.load(f)

        num_node = min(wsi_data['feats'].shape[0], self.maxno)
        features = wsi_data['feats'][:num_node]

        anchor_num = min(wsi_data['knumber'][self.nk_lvl], self.nk)
        anchor_idx = wsi_data['k_idx'][self.nk_lvl]

        rd = wsi_data['rd'][anchor_idx[:anchor_num],:num_node]

        wsi_label = int(self.dl[idx][1][self.ti-1])

        if self.two:
            if self.na:
                data1 = self.random_node_sample(features, rd, num_node)
            else:
                data1 = self.pack_data(features, rd, num_node)
            data2 = self.random_node_sample(features, rd, num_node)

            return data1, data2, wsi_label
        else:
            if self.na:
                data = self.random_node_sample(features, rd, num_node)
            else:
                data = self.pack_data(features, rd, num_node)
                
            return data, wsi_label


    def random_node_sample(self, feat, rd, num_node):
        use_node_idx = np.random.uniform(0.0, 1.0, num_node) > np.random.uniform(0.0, self.ar, 1)
        
        num_node = use_node_idx.sum()
        num_anchor = rd.shape[0]

        wsi_feat = np.zeros((self.maxno, self.feat_dim))
        wsi_rd = np.zeros((self.nk, self.maxno))

        wsi_feat[:num_node] = np.squeeze(feat[use_node_idx])
        wsi_rd[:num_anchor, :num_node] = rd[:,use_node_idx]

        token_mask = np.zeros((self.maxno, 1), int)
        token_mask[:num_node] = 1
        kernel_mask = np.zeros((self.nk, 1), int)
        kernel_mask[:num_anchor] = 1

        return wsi_feat, wsi_rd, token_mask, kernel_mask


    def pack_data(self, feat, rd, num_node):
        num_anchor = rd.shape[0]

        wsi_feat = np.zeros((self.maxno, self.feat_dim))
        wsi_rd = np.zeros((self.nk, self.maxno))

        wsi_feat[:num_node] = np.squeeze(feat)
        wsi_rd[:num_anchor, :num_node] = rd

        token_mask = np.zeros((self.maxno, 1), int)
        token_mask[:num_node] = 1
        kernel_mask = np.zeros((self.nk, 1), int)
        kernel_mask[:num_anchor] = 1

        return wsi_feat, wsi_rd, token_mask, kernel_mask


    def get_wsi_data_path(self, idx):
        return os.path.join(self.list_dir, self.dl[idx][0])

    def get_feat_dim(self):
        return self.feat_dim
        
    def get_weights(self):
        labels = np.asarray([path[1][self.ti-1] for path in self.dl])
        tmp = np.bincount(labels)
        weights = 1 / np.asarray(tmp[labels], np.float)

        return weights


class DistributedWeightedSampler(data.DistributedSampler):
    def __init__(self, dataset, weights, num_replicas=None, rank=None, replacement=True):

        super(DistributedWeightedSampler, self).__init__(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=False
            )
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.replacement = replacement

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.multinomial(self.weights, self.total_size, self.replacement).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


def extract_tile(image_dir, tile_size, x, y, width, height):
    x_start_tile = x // tile_size
    y_start_tile = y // tile_size
    x_end_tile = (x+width) // tile_size
    y_end_tile = (y+height) // tile_size

    tmp_image = np.ones(
        ((y_end_tile-y_start_tile+1)*tile_size, (x_end_tile-x_start_tile+1)*tile_size, 3),
        np.uint8)*240

    for y_id, col in enumerate(range(x_start_tile, x_end_tile + 1)):
        for x_id, row in enumerate(range(y_start_tile, y_end_tile + 1)):
            img_path = os.path.join(image_dir, '{:04d}_{:04d}.jpg'.format(row,col))
            if not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path)
            h, w, _ = img.shape
            tmp_image[(x_id*tile_size):(x_id*tile_size + h), (y_id*tile_size):(y_id*tile_size + w),:] = img

    x_off = x % tile_size
    y_off = y % tile_size
    output = tmp_image[y_off:y_off+height, x_off:x_off+width]
    
    return output


def get_tissue_mask(wsi_thumbnail, scale=30):
    hsv = cv2.cvtColor(wsi_thumbnail, cv2.COLOR_RGB2HSV)
    _, tissue_mask = cv2.threshold(hsv[:, :, 2], 210, 255, cv2.THRESH_BINARY_INV)
    tissue_mask[hsv[:, :, 0]<10]=0

    element = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * scale + 1, 2 * scale + 1)
        )
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, element)

    return tissue_mask


def connectivity_and_dist(positions, down_factor=1):
    power = np.sum(np.multiply(positions, positions), axis=1)
    power = np.repeat(power[np.newaxis, :], positions.shape[0], axis=0)
    relative_dist = np.abs(power - 2*np.dot(positions, np.transpose(positions)) + np.transpose(power))
    adj_mat = relative_dist <= down_factor*down_factor

    return adj_mat, np.sqrt(relative_dist)
