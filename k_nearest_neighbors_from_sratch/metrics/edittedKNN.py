#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Anthony J. Vasquez
email: avasque1@jh.edu
phone: 315-920-8877

Class: Introduction to Machine Learning

Description: Condensed KNN Method
"""
import sys
sys.path.append('../')

##algorithms
from algorithms.kNN import KNN

##metrics
from metrics.metrics import Metrics

class EdittedKNN:
    '''
    Creates Edits the Train set until metrics cease gain
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
    def edit(self):
        metrics_flag = 1 ##flag to stop while loop when Z_Train no longer changes
        used_indices = [] ##keep track of the indices that have already been used
        results_list = [] ##metrics results
        unchanged_cnt = 0
        
        print('X_shape before dropped values: ', self.X_train.shape)
        while metrics_flag:
            ##bookkeeping results
            results_dict = {'truth': [], 'prediction': [], 'k_value': []}
            for index, row in self.X_train.iterrows():
                ##run data through classifier
                ##instantiate classifier model
                classifier = KNN(self.X_train, row, self.k_neighbors,
                                  True, False, self.mode, 
                                  self.max_error, self.sigma)

                ##calculate the distance between test row and train rows
                classifier.distanceCalc()
                ##get the nearest neighbors
                classifier.getNearestNeighbors()

                ##check mode, if mode == regression, then the classifier becomes a regressor object
                if self.mode == 'regression':
                    pred_class, indices_to_drop = classifier.regress()
                else:
                    ##make prediction using the max class
                    pred_class, indices_to_drop = classifier.predictClass()

                if len(indices_to_drop) != 0:
                    for index_to_drop in indices_to_drop:
                        if index_to_drop not in used_indices:
                            used_indices.append(index_to_drop)
                            try:
                                ##drop all rows that the classifier did not correctly classify and if the index exists
                                if len(indices_to_drop) != 0:
                                    for index_to_drop in indices_to_drop:
                                        self.X_train = self.X_train.drop(index=index_to_drop)
                            except:
                                pass
                        else:
                            unchanged_cnt += 1
                            ##if the maximum allowed unchanged occurences are reached, se the metrics flag to zero
                            if unchanged_cnt == self.allowed_unchanged_cnt:
                                ##discontinue train fold
                                metrics_flag = 0
                                # print('Editted_knn: Max allowed unchanged reached', self.allowed_unchanged_cnt)
                            if metrics_flag == 0:
                                break

                         ##record  classifier results in a dictionary
                        results_dict['truth'].append(row[-1])
                        results_dict['prediction'].append(pred_class)
                        results_dict['k_value'].append(self.k_value)

            ##append dictionary to list for input of metrics module
            results_list.append(results_dict)

        ##metrics for no fine tuning case
        if self.mode == 'regression':
            classifier_metrics_obj = Metrics(results_list, self.y_train) ##instantiate classifier
            classifier_metrics_obj.evaluate() ##evaluate results
            ##calculate mean squared error
            mse = classifier_metrics_obj.calculateMSE()
            print('Editted KNN Regression Method')
            print('X Shape after Drop: ', self.X_train.shape)
            print('Train set: ', self.k_value, 'MSE: ',  mse)

        else:
            classifier_metrics_obj = Metrics(results_list, self.y_train) ##instantiate classifier
            classifier_metrics_obj.evaluate() ##evaluate results
            ##calculate average accuracy
            average_accuracy = classifier_metrics_obj.calculateAccuracy()
            print('Editted KNN Classifier Method')
            print('X Shape after Drop: ', self.X_train.shape)
            print('Train set: ', self.k_value, 'Average Accuracy: ',  average_accuracy)
        return self.X_train