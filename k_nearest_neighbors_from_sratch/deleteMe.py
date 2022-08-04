# -*- coding: utf-8 -*-
"""
@author: Anthony J. Vasquez
email: avasque1@jh.edu
phone: 315-920-8877

Class:

Description:
"""
import matplotlib.pyplot as plt
import numpy as np

def get_gauss_kernel(size=3,sigma=1):
    center=(int)(size/2)
    kernel=np.zeros((size,size))
    for i in range(size):
       for j in range(size):
          diff=np.sqrt((i-center)**2+(j-center)**2)
          kernel[i,j]=np.exp(-(diff**2)/(2*sigma**2))
    return kernel/np.sum(kernel)

# gaus_kernal = get_gauss_kernel(100,3)
# plt.imshow(get_gauss_kernel(100,3))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##standard python libraries
import os
import sys
import warnings
import argparse
import time
import gc

import math

import numpy as np
import pandas as pd

##preprocess pipeline
from utils.dataLoaders import LoadCsvData
from utils.preprocess import PreprocessData
from utils.splitData import SplitData
from utils.splitDataClassless import SplitDataClassless
from utils.echoArgs import EchoArgs

##algorithms
from algorithms.kNN import KNN

##metrics
from metrics.metrics import Metrics
from metrics.tuner import Ktuner
from metrics.condensedKNN import CondensedKNN
from metrics.edittedKNN import EdittedKNN

##turn off all warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

gc.collect()

##command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_folder_name', type = str ,default = '/data',
                    help='Name of the folder where the data and names files are located'),

parser.add_argument('--dataset_name', type = str ,default = '/abalone',
                    help='Name of the folder where the data and names files are located'),

parser.add_argument('--namespath', type = str ,default = 'data/abalone.names',
                    help='Path to dataset names'),

parser.add_argument('--discretize_data', type = bool ,default = False,
                    help='Should dataset be discretized?'),

parser.add_argument('--quantization_number', type = int ,default = 3,
                    help='If discretized, then quantization number'),

parser.add_argument('--standardize_data', type = bool , default = True,
                    help='Should data be standardized?'),

parser.add_argument('--k_folds', type = int , default = 5,
                    help='Number of folds for k-fold validation'),

parser.add_argument('--k_neighbors', type = int , default = 6,
                    help='Number of Kernals for KNN'),

parser.add_argument('--min_examples', type = int , default = 15,
                    help='Drop classes with less examples then this value'),

parser.add_argument('--remove_orig_cat_col', type = bool , default = True,
                    help='Remove the original categorical columns for data encoding'),

parser.add_argument('--fine_tune_k', type = bool , default = False,
                    help='fine tune knn model')

parser.add_argument('--knn_values', type = int , default = 10,
                    help='test values of knn up to this number')

parser.add_argument('--fine_tune_error', type = bool , default = True,
                    help='fine tune regression error')

parser.add_argument('--error_values', type = int , default = 10,
                    help='value to fine-tune over for regression')

parser.add_argument('--min_gain', type = float , default = 0.05,
                    help='minimum percentage for each k_neighbor')

parser.add_argument('--editted_knn', type = bool , default = False,
                    help='use edited knn on train/test set ? ')

parser.add_argument('--condensed_knn', type = bool , default = False,
                    help='use condensed knn ? ')

parser.add_argument('--max_error', type = float , default = 50,
                    help='max error for regression to add to drop list')

parser.add_argument('--allowed_unchanged_cnt', type = int , default = 50,
                    help='Number of iteration that the unchanged count can stagnate')

args = parser.parse_args()
## =============================================================================
##                                  MAIN
## =============================================================================
if __name__ == "__main__":
    ##start timer
    tic = time.time()
## =============================================================================
##                              PATHS / ARGUMENTS
## =============================================================================
    ##define paths
    cwd = os.getcwd().replace('\\', '/') ##get current working directory
    data_folder_name = cwd + args.data_folder_name
    datapath = data_folder_name + args.dataset_name + '.data'
    namespath = data_folder_name + args.dataset_name + '.names'
    dataset_name = args.dataset_name

    ##echo argparse arguments
    EchoArgs(data_folder_name, datapath, namespath, args.dataset_name,
             args.discretize_data,args.quantization_number, args.standardize_data,
             args.k_folds, args.k_neighbors,args.min_examples, args.remove_orig_cat_col,
             args.fine_tune_k, args.knn_values, args.min_gain, args.editted_knn,
             args.condensed_knn, args.allowed_unchanged_cnt).echoJob()
## =============================================================================
##                                  PREPROCESS
## =============================================================================
    classification_list = ['car','breast-cancer-wisconsin','house-votes-84']
    regression_list = ['abalone', 'forestfires', 'machine']

    if args.dataset_name.split('/', 1)[1] in classification_list:
        mode = 'classification'
    elif args.dataset_name.split('/', 1)[1] in regression_list:
        mode = 'regression'
    else:
        'NA'

    print('\n******************** ML Pipeline Started ********************')
    ##define tuple of values to drop from dataframe
    values_to_replace = ('na', 'NA', 'nan', 'NaN', 'NAN', '?', ' ')
    values_to_change = {'5more':5, 'more': 5}

    # ##load data
    load_data_obj = LoadCsvData(datapath, namespath, dataset_name)
    names = load_data_obj.loadNamesFromText() ##load names from text
    data = load_data_obj.loadData() ##data to process

    ##preprocess pipeline
    proc_obj = PreprocessData(data, values_to_replace, values_to_change,
                              args.dataset_name, args.discretize_data, args.quantization_number,
                              args.standardize_data, args.remove_orig_cat_col)
    
    proc_obj.dropRowsBasedOnListValues() ##replaces values from list
    proc_obj.changeValues() ##changes values from values_to_change list
    proc_obj.convertDataType() ##converts datatypes of columns based on what they actually are
    proc_obj.replaceValuesFromListWithColumnMean()##replace value with mean
    proc_obj.standardizeData() ##standardizes data
    proc_obj.discretizeData() ##discretizes data
    df_encoded = proc_obj.encodeData() ##encodes data
    
    df_encoded = df_encoded.drop(['target'], axis = 1)
    
    def GaussianKernel(vector_1, vector_2, sigma):
        gaussian_kernal = math.exp(-np.linalg.norm(vector_1-vector_2, 2)**2/(2.*sigma**2))
        return gaussian_kernal
    
    query_vector = df_encoded.iloc[0].values
    for i in range(len(df_encoded['length'])):
        v = df_encoded.iloc[i].values
        gk = GaussianKernel(query_vector, v, 5)
        print(gk)
        
    
    
    
    