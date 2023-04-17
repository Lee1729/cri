# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from glob import glob
import os
import graphviz
from sklearn import tree as skt
import argparse
import sys

workingdir = os.path.join(os.getcwd(), '..')
os.chdir(workingdir + '/modules' )
sys.path.extend([os.path.abspath(".")])
from usertree import userTree as utr
import utils

"""
###########################
Set parameters
"""
parser = argparse.ArgumentParser(description='')
parser.add_argument('--n_parts', dest='n_parts', type=int, default=6,\
                    help='The number of task performed independently.\
                    = max value of part_num in ADAPTIVE_with_BASEMODEL_train.py')
parser.add_argument('--save_dir', dest='save_dir', default= 'experiment_X',\
                    help='save dir')
parser.add_argument('--iter_', dest='iter_', type=int, default=5,\
                    help='Number of measurement iteration.')

args, _ = parser.parse_known_args()

n_parts = args.n_parts
save_search_dir = args.save_dir
iteration = args.iter_

os.chdir(os.path.join(workingdir, save_search_dir))
d_dir = os.path.join(workingdir, 'dataset')
load_models_dir = 'models'

# target attribute name
target_att = 'target'

"""
###########################
data load & set save dir
"""
dataset_dir = glob(d_dir + '/*.csv')
data_diction = {dset.split('\\')[-1][:-4]: pd.read_csv(dset) \
                for dset in dataset_dir}
dataset_list = sorted(list(data_diction.keys()), reverse=True)

perform_train = []
perform_test= []

for d_set in dataset_list[10:12]:
    for iter_ in range(iteration):
        for part_ in range(n_parts):
            print('======================\ndataset name : ' + d_set )
            #load model & data
            model_dir = '{}_{}_{}'.format(d_set, iter_, part_)
            model_dict = utils.load_obj(os.path.join(load_models_dir, model_dir))
            data = data_diction[d_set]
            data = data.reset_index(drop=True)
            data.target =data.target
            colnm = data.columns
            X = data.loc[:,colnm [colnm != target_att]]
            y = data.loc[:, target_att]

            in_feature = list(data.columns[data.columns != target_att])

            cate_col = [col for col in in_feature \
                        if not np.issubdtype(X[col ].dtype, \
                np.number)]
            X_dummies = pd.get_dummies(data.loc[:,in_feature], columns=cate_col)
            CLASSES_LIST = np.unique(data[target_att].values)

            #trained model
            [NEW_tree_ins] = model_dict[('New_Tree')]
            NEW_tree, NEW_graph_tree = \
                NEW_tree_ins.tree, NEW_tree_ins.graph_tree

            # get train, test data set
            tree_rule = utils.get_leaf_rule(NEW_tree, [], [], leaf_info=True)
            if tree_rule[0]=='Root_node':
                train_idx = set(tree_rule[-1][1].index)
            else:
                train_idx = set(sum([ list(rule[-1][-1].index) \
                                    for rule in tree_rule],[]))
            test_idx = set(data.index) - train_idx
            train, test= data.loc[train_idx], data.loc[test_idx]

            """
            # measure performance
            """
            classes = [str(NEW_tree_ins.CLASS_DICT[i]) for i in range(NEW_tree_ins.NUM_CLASSES)]

            #NEW
            train_NEW_all_pred, train_NEW_all_pred_prob = NEW_tree_ins.predict(train, NEW_tree)
            test_NEW_all_pred, test_NEW_all_pred_prob = NEW_tree_ins.predict(test, NEW_tree)

            train_NEW_all_mac = utils.perform_check(train['target'], \
                         train_NEW_all_pred, train_NEW_all_pred_prob, \
                         len(classes), NEW_tree_ins.CLASS_DICT_, average = 'macro')
            test_NEW_all_mac = utils.perform_check(test['target'], \
                         test_NEW_all_pred, test_NEW_all_pred_prob, \
                         len(classes), NEW_tree_ins.CLASS_DICT_, average = 'macro')

            #NEW
            train_NEW_mac_ACC, _, _, \
            train_NEW_mac_F1, train_NEW_mac_AUC = \
                train_NEW_all_mac

            perform_train.append([iter_, part_, d_set,\
                        train_NEW_mac_ACC, \
                        train_NEW_mac_F1])

            ## test
            #NEW
            test_NEW_mac_ACC, _, _, \
            test_NEW_mac_F1, test_NEW_mac_AUC = \
                test_NEW_all_mac

            perform_test.append([iter_, part_, d_set,\
                        test_NEW_mac_ACC, \
                        test_NEW_mac_F1])


            print('ACCURACY', iter_, part_, d_set, np.round(np.round(test_NEW_mac_ACC, 3)))

perform_train_df = pd.DataFrame(perform_train, columns = \
    ['iteration', 'part', 'dataset_name',\
    'NEW_mac_ACC', \
    'NEW_mac_F1'])

perform_test_df = pd.DataFrame(perform_test, columns = \
    ['iteration', 'part', 'dataset_name',\
    'NEW_mac_ACC', \
    'NEW_mac_F1'])

perform_train_df.to_csv('performance_train.csv', index=False)
perform_test_df.to_csv('performance_test.csv', index=False)

"""
############################
Summary
############################
"""

#load
perform_test_all_df = pd.read_csv('performance_test.csv')
perform_test_all_df = pd.read_csv('performance_test.csv')

writer = pd.ExcelWriter('Results.xlsx')

### perform_testance
# average ------------------------------------------------------
avg_perform_test1 = perform_test_all_df.loc[:,['dataset_name', \
    'NEW_mac_ACC', 'NEW_mac_F1']].\
    groupby(['dataset_name'], as_index =False).mean()
avg_perform_test1['criteria'] = 'NEW'

avg_perform_test = pd.concat([avg_perform_test1], axis=0)
avg_perform_test.set_index(['dataset_name','criteria'],inplace=True)
avg_perform_test.columns = [ 'TEST_MACRO_ACC', 'TEST_MACRO_F1']

summary_avg_perform_test = avg_perform_test .unstack(level=1)

temp = avg_perform_test.reset_index()
temp123 = temp.loc[:, ['dataset_name', 'criteria', \
                'TEST_MACRO_ACC', 'TEST_MACRO_F1']\
        ].groupby(['criteria'], as_index =False).mean()
for c in [ 'TEST_MACRO_ACC', 'TEST_MACRO_F1']:
    summary_avg_perform_test.loc['AVERAGE', c] = temp123.loc[:,c].values

summary_avg_perform_test = np.round(summary_avg_perform_test, 3)

# summary
row, col = summary_avg_perform_test.shape
summary_perform_test= summary_avg_perform_test.copy()
for r in range(row):
    for c in range(col):
        summary_perform_test.iloc[r,c] = \
                '{}'.format(summary_avg_perform_test.iloc[r,c])

summary_perform_test.to_excel(writer, 'performace_test')
summary_avg_perform_test.to_excel(writer, 'avg_performace_test')
writer.save()
