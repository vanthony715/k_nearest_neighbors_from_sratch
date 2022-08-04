#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Anthony J. Vasquez
email: avasque1@jh.edu
phone: 315-920-8877

Class: Introduction to Machine Learning

Description: Project 1 - Main
"""

##standard python libraries
import os
import sys
import warnings
import argparse
import time
import gc

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

'''
Datasets:
    classification: 'car','breast-cancer-wisconsin','house-votes-84'
    regression_list: 'abalone', 'forestfires', 'machine'
'''

##command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_folder_name', type = str ,default = '/data',
                    help='Name of the folder where the data and names files are located'),

parser.add_argument('--dataset_name', type = str ,default = '/abalone',
                    help='Name of the folder where the data and names files are located'),

parser.add_argument('--namespath', type = str , default = 'data/abalone',
                    help='Path to dataset names'),

parser.add_argument('--discretize_data', type = bool ,default = False,
                    help='Should dataset be discretized?'),

parser.add_argument('--quantization_number', type = int ,default = 5,
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

parser.add_argument('--fine_tune_error', type = bool , default = False,
                    help='fine tune regression error')

parser.add_argument('--error_values', type = int , default = 10,
                    help='value to fine-tune over for regression')

parser.add_argument('--min_gain', type = float , default = 0.05,
                    help='minimum percentage for each k_neighbor')

parser.add_argument('--editted_knn', type = bool , default = False,
                    help='use edited knn on train/test set ? ')

parser.add_argument('--condensed_knn', type = bool , default = True,
                    help='use condensed knn ? ')

parser.add_argument('--max_error', type = float , default = 50,
                    help='max error for regression to add to drop list')

parser.add_argument('--allowed_unchanged_cnt', type = int , default = 50,
                    help='Number of iteration that the unchanged count can stagnate')

parser.add_argument('--sigma', type = int , default = 5,
                    help='Gaussian Kernal Sigma for Regression')

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
## =============================================================================
##                                  PREPROCESS
## ============================================================================

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
    
    

    ##this is the only dataset that needs particularly special attention
    if args.dataset_name.split('/', 1)[1] == 'machine':
        split_obj = SplitDataClassless(df_encoded, args.k_folds, args.min_examples)
        split_obj.splitPipeline() ##start of the stratefied k-fold validation split
        train_test_sets = split_obj.createTrainSets() ##k train and test sets returned as a dictionary
        train_columns = split_obj.getTrainColumns() ##gets all columns but target
    else:
        split_obj = SplitData(df_encoded, args.k_folds, args.min_examples)
        split_obj.removeSparseClasses() ##removes classes that do not meet the min_examples criteria
        split_obj.countDataClasses() ##counts data classes
        split_obj.splitPipeline() ##start of the stratefied k-fold validation split
        train_test_sets = split_obj.createTrainSets() ##k train and test sets returned as a dictionary
        train_columns = split_obj.getTrainColumns() ##gets all columns but target

## =============================================================================
##                                  TRAIN
## =============================================================================
    #instantiate list
    results_list = []

    ##instantiate classifier and regressor and trains/tests each set
    avg_accuracy_dict = {'avg_accuracy': [], 'k_neighbor': [], 'k_value': []}
    for k in range(len(train_test_sets['train_set'])):
        ##get train data and labels
        X_train = train_test_sets['train_set'][k].loc[:, train_columns]
        y_train = train_test_sets['train_set'][k]['target']

        ##get test data and labels
        X_test = train_test_sets['test_set'][k].loc[:, train_columns]
        y_test = train_test_sets['test_set'][k]['target']

        ##indicate the train test iteration
        print('\nTrain/Test Set: ', k)

        ##instantiate naive classifier model
        X_train['target'] = y_train ##combine labels and data for knn
        X_test['target'] = y_test

        ##Editted KNN Method
        if args.editted_knn:
            ##instantiate Editted knn model
            editted_knn_obj = EdittedKNN(X_train, y_train, args.allowed_unchanged_cnt,
                                          args.k_neighbors, k, mode, 
                                          args.sigma, args.max_error)
            editted_X_train = editted_knn_obj.edit() ##execute editted knn

        ##condensed KNN Method
        elif args.condensed_knn:
            ##instantiate condensed knn model
            condensed_knn_obj = CondensedKNN(X_train, y_train, args.allowed_unchanged_cnt,
                                              args.k_neighbors, k, mode, 
                                              args.sigma, args.max_error)
            condensed_knn_obj.condense() ##execute editted_knn

        ##finetune k nearest neighbors
        elif args.fine_tune_k or args.fine_tune_error:
            ##create k_nearest neighbors values to iterate through as candidates
            k_neighbors_array = np.arange(args.knn_values)

            ##tune model
            tuner_obj = Ktuner(X_train, y_train, X_test, k, k_neighbors_array,
                                args.min_gain, False, False, mode, args.max_error,
                                args.error_values, args.fine_tune_k, 
                                args.fine_tune_error, args.sigma) ##instantiate tuner

            ## get max accuracy and associate knn value
            tuner_obj.tune()

        ##standard cross-validation KNN
        else:
            print('crossValidation')
            results_dict = {'truth': [], 'prediction': [], 'k_value': []}
            for index, row in X_train.iterrows():
                ##instantiate classifier model
                classifier = KNN(X_train, row, args.k_neighbors,
                                  False, False, mode, 
                                  args.max_error, args.sigma)
                
                classifier.distanceCalc()
                
                ##get the nearest neighbors
                classifier.getNearestNeighbors()

                ##check mode, if mode == regression, then the classifier becomes a regressor object
                if mode == 'regression':
                    pred_class, indices_to_drop = classifier.regress()
                else:
                    ##make prediction using the max class
                    pred_class, indices_to_drop = classifier.predictClass()

                ##record classifier results in a dictionary
                results_dict['truth'].append(row[-1])
                results_dict['prediction'].append(pred_class)
                results_dict['k_value'].append(k)

            ##append dictionary to list for input of metrics module
            results_list.append(results_dict)

            ##metrics for no fine tuning case
            if mode == 'regression':
                classifier_metrics_obj = Metrics(results_list, y_train) ##instantiate classifier
                metrics_dict = classifier_metrics_obj.evaluate() ##evaluate results

                ##calculate mean squared error
                mse = classifier_metrics_obj.calculateMSE()
                print('Standard KNN Regression Method')
                print('Train set: ', k, 'MSE: ',  mse)

            else:
                classifier_metrics_obj = Metrics(results_list, y_train) ##instantiate classifier
                metrics_dict = classifier_metrics_obj.evaluate() ##evaluate results
                ##calculate average accuracy
                average_accuracy = classifier_metrics_obj.calculateAccuracy()
                print('Standard KNN Classifier Method')
                print('Train set: ', k, 'Average Accuracy: ',  average_accuracy)

    toc = time.time()
    tf = round((toc - tic), 2)
    print('Total Time: ', tf)