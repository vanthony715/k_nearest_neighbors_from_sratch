#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Anthony J. Vasquez
email: avasque1@jh.edu
phone: 315-920-8877

Class: Machine Learning

Description: Echos argparse arguments
"""

class EchoArgs:

    '''
    Echos Arguments
    '''
    def __init__(self, data_folder_name, datapath, namespath, dataset_name, discretize_data,
                 quantization_number, standardize_data, k_folds, k_neighbors, min_examples,
                 remove_orig_cat_col, fine_tune_k, knn_values, min_gain, editted_knn,
                 condensed_knn, allowed_unchanged_cnt):

        self.data_folder_name = data_folder_name
        self.datapath = datapath
        self.namespath = namespath
        self.dataset_name = dataset_name
        self.discretize_data = discretize_data
        self.quantization_number = quantization_number
        self.standardize_data = standardize_data
        self.k_folds = k_folds
        self.k_neighbors = k_neighbors
        self.min_examples = min_examples
        self.remove_orig_cat_col = remove_orig_cat_col
        self.fine_tune_k = fine_tune_k
        self.knn_values = knn_values
        self.min_gain = min_gain
        self.editted_knn = editted_knn
        self.condensed_knn = condensed_knn
        self.allowed_unchanged_cnt = allowed_unchanged_cnt

    def echoJob(self):
        print('\n\n\n----------------------------------------------------------')
        print('\n--------------------- Job Description --------------------')
        print('Data folder name: ', self.data_folder_name)
        print('Path to data: ', self.datapath)
        print('Names file name: ', self.namespath)
        print('Dataset name: ', self.dataset_name)
        print('Discretize data?: ', self.discretize_data)
        print('Quantization number: ', self.quantization_number)
        print('Standardize data: ', self.standardize_data)
        print('K-Folds: ', self.k_folds)
        print('K-Nearest Neighbors: ', self.k_neighbors)
        print('Min number of examples: ', self.min_examples)
        print('Remove original cat column when decoding?: ', self.remove_orig_cat_col)
        print('Finetuning: ', self.fine_tune_k)
        print('K-nearest neighbors: ', self.knn_values)
        print('Editted_Knn: Minimum gain expected: ', self.min_gain)
        print('Editted KNN Method: ', self.editted_knn)
        print('Condensed KNN Method: ', self.condensed_knn)
        print('Condensed KNN: Maximum Iterations without gain: ', self.condensed_knn)
        print('----------------------------------------------------------')
        print('----------------------------------------------------------\n')