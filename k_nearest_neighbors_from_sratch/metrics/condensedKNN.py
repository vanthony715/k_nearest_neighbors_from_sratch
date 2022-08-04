#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Anthony J. Vasquez
email: avasque1@jh.edu
phone: 315-920-8877

Class: Introduction to Machine Learning

Description: Condensed KNN Method
"""

import pandas as pd

##algorithms
from algorithms.kNN import KNN

##metrics
from metrics.metrics import Metrics

class CondensedKNN:
    '''
    Creates Z and stops when length Z is unchanging
    '''
    def __init__(self, X_train, y_train, allowed_unchanged_cnt, k_neighbors,
                 k_value, mode, max_error, sigma):
        self.X_train = X_train
        self.y_train = y_train
        self.allowed_unchanged_cnt = allowed_unchanged_cnt
        self.k_neighbors = k_neighbors
        self.k_value = k_value
        self.mode = mode
        self.max_error = max_error
        self.sigma = sigma

    ##adds to Z_train and drops from X_train
    def condense(self):
        metrics_flag = 1 ##flag to stop while loop when Z_Train no longer changes
        used_indices = [] ##keep track of the indices that have already been used
        results_list = [] ##metrics results
        unchanged_cnt = 0
        Z_train = pd.DataFrame(columns = list(self.X_train))
        
        print('Z Shape before add: ', Z_train.shape)
        while metrics_flag:
            ##bookkeeping results
            results_dict = {'truth': [], 'prediction': [], 'k_value': []}
            for index, row in self.X_train.iterrows():
                ##run data through classifier
                ##instantiate classifier model
                classifier = KNN(self.X_train, row, self.k_neighbors,
                                  False, True, self.mode, self.max_error,
                                  self.sigma)

                ##calculate the distance between test row and train rows
                classifier.distanceCalc()
                ##get the nearest neighbors
                classifier.getNearestNeighbors()

                ##check mode, if mode == regression, then the classifier becomes a regressor object
                if self.mode == 'regression':
                    pred_class, indices_to_add = classifier.regress()
                else:
                    ##make prediction using the max class
                    pred_class, indices_to_add = classifier.predictClass()
                
                if len(indices_to_add) != 0:
                    for index_to_add in indices_to_add:
                        if index_to_add not in used_indices:
                            used_indices.append(index_to_add)
                            ##add the succesful classifications to Z_train
                            try:
                                ##get row to add to Z
                                row_to_add = self.X_train.iloc[[index_to_add]]
                                Z_train = Z_train.append(row_to_add)

                                ##drop the row from Z_train
                                self.X_train = self.X_train.drop(index=index_to_add)
                            except:
                                pass
                        else:
                            unchanged_cnt += 1
                            ##if the maximum allowed unchanged occurences are reached, se the metrics flag to zero
                            if unchanged_cnt == self.allowed_unchanged_cnt:
                                ##discontinue train fold
                                metrics_flag = 0
                                # print('Max allowed unchanged Z reached', self.allowed_unchanged_cnt)
                            if metrics_flag == 0:
                                break

                         ##record classifier results in a dictionary
                        results_dict['truth'].append(row[-1])
                        results_dict['prediction'].append(pred_class)
                        results_dict['k_value'].append(self.k_value)

           ##append dictionary to list for input of metrics module
            results_list.append(results_dict)

        ##metrics for no fine tuning case
        if self.mode == 'regression':
            classifier_metrics_obj = Metrics(results_list, self.y_train) ##instantiate classifier
            metrics_dict = classifier_metrics_obj.evaluate() ##evaluate results
            ##calculate mean squared error
            mse = classifier_metrics_obj.calculateMSE()
            print('Codensed KNN Regression Method')
            print('Z Shape after add: ', Z_train.shape)
            print('Train set: ', self.k_value, 'MSE: ',  mse)

        else:
            classifier_metrics_obj = Metrics(results_list, self.y_train) ##instantiate classifier
            metrics_dict = classifier_metrics_obj.evaluate() ##evaluate results
            ##calculate average accuracy
            average_accuracy = classifier_metrics_obj.calculateAccuracy()
            print('Condensed KNN Classifier Method')
            print('Z Shape after add: ', Z_train.shape)
            print('Train set: ', self.k_value, 'Average Accuracy: ',  average_accuracy)
        return metrics_dict