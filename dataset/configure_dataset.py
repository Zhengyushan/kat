#!/usr/bin/env python
# -*- coding:utf-8 -*-
# date: 2022/06
# author:Yushan Zheng
# emai:yszheng@buaa.edu.cn

import pickle
import os

'''
The script to configure the dataset for running the code.
Take these TCGA Slides as the example.
You may need to these Slides to the experimental files. 
Please refer to https://github.com/zhengyushan/tcga2tile.
'''

# TCGA Lung dataset
data_list_dir = 'tcga_lung'

# Make slide list
train_slide_list = []
with open(os.path.join(data_list_dir, 'train.csv'), 'r') as f:
    line = f.readline()
    line = f.readline()
    while(len(line) > 0):
        guid, rpath, label = line.split(',')
        output_line = [guid, rpath, int(label)]
        train_slide_list.append(output_line)
        line = f.readline()

test_slide_list = []
with open(os.path.join(data_list_dir, 'test.csv'), 'r') as f:
    line = f.readline()
    line = f.readline()
    while(len(line) > 0):
        guid, rpath, label = line.split(',')
        output_line = [guid, rpath, int(label)]
        test_slide_list.append(output_line)
        line = f.readline()

with open(os.path.join(data_list_dir, 'slide_list.pkl'), 'wb') as f:
    pickle.dump(
        {
            'train':train_slide_list,
            'test':test_slide_list
        }, f
    )

# Make datset config
lesions = ["Normal", "A.", "SCN",]

tasks = {
    1:{'num_classes':3, 'map':{0:0, 1:1, 2:2}},
    2:{'num_classes':2, 'map':{0:0, 1:1, 2:1}},
}

with open('tcga_lung/dataset_config.pkl', 'wb') as f:
    pickle.dump(
        {'tasks':tasks,
        'lesions':lesions,
        }, f
    )