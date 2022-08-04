# -*- coding: utf-8 -*-
"""
@author: Anthony J. Vasquez
email: avasque1@jh.edu
phone: 315-920-8877

Class: Introduction to Machine Learning

Description: All functions needed to preprocess data
"""
import pandas as pd
import numpy as np

class PreprocessData:
    '''
    Preprocess dataframe
    '''
    def __init__(self, data, values_to_replace, values_to_change, dataset_name,
                 discretize_data, quantization_number, standardize_data, remove_orig_cat_col):
        print('Preprocess data initialized')
        self.data = data
        self.values_to_replace = values_to_replace ##list of values to replace
        self.values_to_change = values_to_change ##list of values to change Ex. 5more -> 5
        self.dataset_name = dataset_name ## name of dataset
        self.discretize_data = discretize_data ##discretize data flag
        self.quantization_number = quantization_number ##number to quantize data to
        self.standardize_data = standardize_data ##standardize data flag
        self.remove_orig_cat_col = remove_orig_cat_col ##remove the original categorical data

    ##drop rows containing cells with values values_to_drop
    def dropRowsBasedOnListValues(self):
        self.col_names = list(self.data)
        for col_name in self.col_names:
            self.data = self.data[self.data[col_name].isin(self.values_to_replace) == False]
        self.data = self.data.reset_index(drop=True) ##reset index for any dropped rows
        return self.data

    def changeValues(self):
        keys = list(self.values_to_change)
        for col_name in self.col_names:
            for key in keys:
                self.data.loc[self.data[col_name] == key, col_name] = self.values_to_change[key]
        return self.data

    def convertDataType(self):
        ##convert target to int if a float
        if isinstance(self.data['target'][0], np.float64) or isinstance(self.data['target'][0], np.int64):
            for idx in range(len(self.data['target'])):
                self.data['target'][idx] = int(self.data['target'][idx])
        ##define number strings to test against
        number_list = np.arange(150) ##this seems to be a good value after inspection of datasets
        str_number_list = [str(i) for i in number_list]
        self.category_list = []
        for col_name in self.col_names: #check each column
            flag = None # reset flag
            # col_data = list(self.data[self.col_name])
            search_length = 150 #how far to search in dataframe column for misleading datatypes

            ##kickoff float, int, categorical
            for i in range(1, search_length):
                value = self.data[col_name][i]

                ##first check if the value is a float
                if isinstance(value, np.float64):
                    flag = 'float' ## passes
                    self.data[col_name] = self.data[col_name].astype(flag)
                    break

                ##check if value is a string that is actually a float
                if isinstance(value, str):
                    try: ##test value is float
                        float_test = value.split('.')[1] #shall be split by decimal
                        if float_test in str_number_list or float_test in number_list: ##establishes float_test as a number
                            flag = 'float' ## passes
                            self.data[col_name] = self.data[col_name].astype(flag)
                            break
                    except:
                        pass #fails float test

                if isinstance(value, np.int64) or isinstance(value, np.int32):
                    flag = 'int' ## passes
                    self.data[col_name] = self.data[col_name].astype(np.int64)
                    break

                elif isinstance(value, str) and value in str_number_list: ##establishes value is an int
                    flag = 'int' ## passes
                    self.data[col_name] = self.data[col_name].astype(np.int64)
                    break

                ##must be categorical
                elif flag != 'float' and flag != 'int':
                    flag = 'category'
                    self.data[col_name] = pd.Categorical(self.data[col_name])
                    self.category_list.append(col_name)
            # print('column1: ', self.col_name, 'flag:', self.data[self.col_name].dtype)
        return self.data

    ##replace containing cells with values values_to_drop
    def replaceValuesFromListWithColumnMean(self):
        for col_name in self.col_names:
            if col_name not in self.category_list:
                for value in self.values_to_replace:
                    self.data.loc[self.data[col_name] == value , col_name] = self.data[col_name].mean()
        return self.data

    ##replace containing cells with values values_to_drop
    def encodeData(self):
        for col_name in self.col_names:
            if col_name in self.category_list:
                if col_name != 'target':
                    dummies = pd.get_dummies(self.data[col_name], drop_first=True)
                    if self.remove_orig_cat_col:
                        self.data = self.data.drop([col_name], axis=1) ##delete original categorical data
                        print('Removed Original Categorical Column: ', col_name)
                    else:
                        self.data = pd.concat([self.data, dummies], axis = 1)
                        print('Encoded Column: ', col_name)
        return self.data

    def discretizeData(self):
        if self.discretize_data:
            for col_name in self.col_names:
                if col_name not in self.category_list:
                    self.data[col_name] = pd.qcut(self.data[col_name], q=self.quantization_number)
                    print('Quantized: ', col_name)
        return self.data

    def standardizeData(self):
        if self.standardize_data:
            for col_name in self.col_names:
                if col_name not in self.category_list and col_name != 'target':
                    column_mean = self.data[col_name].mean()
                    column_std = self.data[col_name].std()
                    for i in range(len(self.data[col_name])):
                        z = (self.data[col_name][i] - column_mean) / column_std
                        self.data[col_name][i] = z
        return self.data