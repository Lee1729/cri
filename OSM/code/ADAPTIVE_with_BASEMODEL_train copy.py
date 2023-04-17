# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from time import time
from glob import glob
import os
from sklearn.tree import DecisionTreeClassifier as sk_cart
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import sys
import argparse

workingdir = os.path.join(os.getcwd(), '..')
os.chdir(os.path.join(workingdir, 'modules' ))
sys.path.extend([os.path.abspath(".")])
from usertree import userTree as utr
import utils
plt.ioff()

"""
-------------------------------------------------------------------------------------------
Set parameters
"""
parser = argparse.ArgumentParser(description='')
parser.add_argument('--part_num', dest='part_num', type=int, default=1,\
                    help='It is an integer for distributing work. \
                    The following operations are performed independently \
                    and stored by the given number.')
parser.add_argument('--save_dir', dest='save_dir', default= 'experiment_X',\
                    help='save dir')
parser.add_argument('--iter_', dest='iter_', type=int, default=5,\
                    help='Number of measurement iteration.')

args, _ = parser.parse_known_args()

part_num = args.part_num
save_dir = args.save_dir
test_iteration = args.iter_

# target attribute name
target_att = 'target'

# train(train:test) : test = 8(8:2) : 2
test_ratio = 0.2

# NEW DT params
lambda_range = sorted([1-np.log10(i) for i in np.arange(1,10,1)])

# termination criteria
MAX_DEPTH = 1000
sample_ratio = 0.01

os.chdir(os.path.join(workingdir))
"""
########################################################################
data load & set save dir
"""
dataset_dir = glob(os.path.join(workingdir, 'dataset', '*.csv'))
data_diction = {dset.split('\\')[-1][:-4]: pd.read_csv(dset) for dset in dataset_dir}
dataset_list = sorted(list(data_diction.keys()), reverse=True)

#for save_dir in save_dir_list:
save_fig_dir = os.path.join(save_dir, 'lambda_plot')
model_save_dir = os.path.join(save_dir, 'models')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(save_fig_dir):
    os.makedirs(save_fig_dir)
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

"""
-----------------------------------------------------------------------------
Fitting
"""
for d_set in dataset_list[11:12]: #dataset_list:
    data = data_diction[d_set]
    data = data.reset_index(drop=True)
    colnm = data.columns
    X = data.loc[:,colnm [colnm != target_att]]
    y = data.loc[:, target_att]
    target_elements = np.unique(y)

    n_samples = round(sample_ratio * len(data))
    in_feature = list(data.columns [data.columns != target_att])

    CLASSES_LIST = np.unique(data[target_att].values)

    model_dict = {}
    for iter_ in range(test_iteration):
        X_train, X_test, _, _ = train_test_split(X, y, test_size=test_ratio, \
                stratify=y, shuffle =True)
        train_idx, test_idx = X_train.index, X_test.index
        train, test= data.loc[train_idx,:], data.loc[test_idx,:]

        cate_col = [col for col in in_feature if not np.issubdtype(X[col].dtype, \
            np.number)]
        X_dummies= pd.get_dummies(data.loc[:,in_feature], columns=cate_col)

        print('===================\n{}, {}, {}\n--------------------'.\
             format(d_set, len(data), iter_))

        # OSM
        new_tree_ins = utr(n_samples, MAX_DEPTH, params=lambda_range, split_algo='adaptive')

        print('start NEW')
        st = time()
        max_tree, max_pprint_tree = new_tree_ins.fit(train, target_attribute_name = "target")
        et = time()
        max_tree_time = et - st
        model_dict[('New_Tree')] = [new_tree_ins]

        print('traing_time, NEW:{}'.format(round(max_tree_time,3)))

        save_models_name = model_save_dir + '/{}_{}_{}'.format(d_set, iter_, part_num)
        #import ipdb; ipdb.set_trace()
        utils.save_obj(model_dict, save_models_name)