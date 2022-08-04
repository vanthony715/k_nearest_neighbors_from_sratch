#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Anthony J. Vasquez
email: avasque1@jh.edu
phone: 315-920-8877

Class: Introduction to Machine Learning

Description: Tune k value
"""

from algorithms.kNN import KNN
from metrics.metrics import Metrics
class Ktuner(KNN):
    '''
    Tunes a model's k value to improve model performance
    '''
    def __init__(self, X_train, y_train, X_test, k_value, knn_array, min_gain,
                 condensed_knn, editted_knn, mode, max_error, error_array,
                 finetune_k, finetune_e, sigma):

        self.X_train = X_train ##train data
        self.X_test = X_test ##test labels
        self.y_train = y_train ##train labels
        self.k_value = k_value ##k-fold
        self.knn_array = knn_array ##values of k for tuning
        self.min_gain = min_gain ##minimum accuracy gain between k_neighbor value
        self.condensed_knn = condensed_knn ##perform condensed knn
        self.editted_knn = editted_knn ##perform editted_knn
        self.mode = mode ##either classification or linear regression
        self.max_error = max_error ##cut off error value for regression
        self.error_array = error_array##error values for tuning
        self.finetune_k = finetune_k ##should k be tuned
        self.finetune_e = finetune_e ##should error be tuned
        self.sigma = sigma ##finetune sigma

    def tune(self):
        if self.finetune_k and self.mode == 'classification':
            results_list = [] ##record results as input to metrics module
            ##keep track of all accuracies for each k fold
            avg_accuracy_dict = {'avg_accuracy': [], 'k_neighbor': [], 'k_value': []}
            for i in range(1, len(self.knn_array)):

                ##keep track of results output by classifier
                results_dict = {'truth': [], 'prediction': [], 'k_value':[]}
                ##iterate through test set
                for index, row in self.X_test.iterrows():
                    ##classifier
                    classifier = KNN(self.X_train, row, i, self.condensed_knn,
                                     self.editted_knn, self.mode, self.max_error,
                                     self.sigma) ##instantiate classifier
                    classifier.distanceCalc() ##calculate distance between rows
                    classifier.getNearestNeighbors() ##get the nearest neighbors
                    pred_class, indices_to_drop = classifier.predictClass() ##predict the class

                    ##append results to dictionary
                    results_dict['truth'].append(row[-1])
                    results_dict['prediction'].append(pred_class)
                    results_dict['k_value'].append(self.k_value)

                ##append dictionary to a list as input for metrics module
                results_list.append(results_dict)

                ##get metrics
                classifier_metrics_obj = Metrics(results_list, self.y_train) ##instantiate metrics
                classifier_metrics_obj.evaluate() ##evaluate test set
                average_accuracy = classifier_metrics_obj.calculateAccuracy() ##calculate the accuracy

                ##append accuracy to dictionary
                print('K_neighbors: ', i, 'average accuracy: ', average_accuracy)
                avg_accuracy_dict['k_neighbor'].append(i)
                avg_accuracy_dict['avg_accuracy'].append(average_accuracy)

                ##check that there are atleast two values in dictionary list
                if i >= 10:
                    ##check percentage change
                    percentage_change = 1 - avg_accuracy_dict['avg_accuracy'][i - 2] / avg_accuracy_dict['avg_accuracy'][i - 1]

                    ##compare percentage change to min allowed gain, if less than break
                    if percentage_change < self.min_gain:
                        break

            ##get the max accuracy and associated knn value for tune output
            max_accuracy = max(avg_accuracy_dict['avg_accuracy'])
            max_index = avg_accuracy_dict['avg_accuracy'].index(max_accuracy)
            max_k_neighbor = avg_accuracy_dict['k_neighbor'][max_index]

            ##append values to dictionary for later use in max accuracy step
            avg_accuracy_dict['avg_accuracy'].append(max_accuracy)
            avg_accuracy_dict['k_neighbor'].append(max_k_neighbor)
            avg_accuracy_dict['k_value'].append(self.k_value)

            print('Fine-tune KNN Method')
            print('max accuracy of: ', max_accuracy, 'at k nearest neighbor: ', max_k_neighbor)

        elif not self.finetune_k and self.finetune_e and self.mode == 'regression':
            results_list = [] ##record results as input to metrics module
            ##keep track of all accuracies for each k fold
            for e in range(1, self.error_array):
                print('Fine-tune Error Method')
                ##keep track of results output by classifier
                results_dict = {'truth': [], 'prediction': [], 'k_value':[]}
                ##iterate through test set
                for index, row in self.X_test.iterrows():
                    ##classifier
                    regressor = KNN(self.X_train, row, e, self.condensed_knn,
                                     self.editted_knn, self.mode, self.max_error,
                                     self.sigma) ##instantiate classifier
                    regressor.distanceCalc() ##calculate distance between rows
                    regressor.getNearestNeighbors() ##get the nearest neighbors
                    prediction, indices_to_drop = regressor.regress() ##predict the class
                    
                    ##append results to dictionary
                    results_dict['truth'].append(row[-1])
                    results_dict['prediction'].append(prediction)
                    results_dict['k_value'].append(self.k_value)

                ##append dictionary to a list as input for metrics module
                results_list.append(results_dict)

                ##get metrics
                regressor_metrics_obj = Metrics(results_list, self.y_train) ##instantiate metrics
                regressor_metrics_obj.evaluate() ##evaluate test set
                mse_dict = regressor_metrics_obj.calculateMSE() ##calculate the accuracy

                ##append accuracy to dictionary
                results_list.append(mse_dict)