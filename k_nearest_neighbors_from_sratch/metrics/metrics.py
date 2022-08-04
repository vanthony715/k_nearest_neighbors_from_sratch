#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Anthony J. Vasquez
email: avasque1@jh.edu
phone: 315-920-8877

Class: Introduction to Machine Learning

Description: Computes Metrics
"""
import random
import pandas as pd
import numpy as np

class Metrics:
    '''
    Calculates the metrics from test output
    '''
    def __init__(self, results_list, y_train):
        print('\n********************* Results ********************')
        print('\nMetrics Initialized')
        self.results_list = results_list
        self.y_train = y_train

    ##this function compares the predictions to the truth values and assigns a 1 for correct or 0 for incorrect
    ##this is only used for classification
    def evaluate(self):
        self.metrics_dict = {'target':[], 'correct': [], 'incorrect': [], 'total': []}
        k_values = list(set(self.results_list[0]['k_value']))
        classes = list(set(self.results_list[0]['truth']))
        for clss in classes:
            total_cnt = 0
            correct_class_cnt = 0
            incorrect_class_cnt = 0
            for results_dict in self.results_list:
                for i, value in enumerate(results_dict['truth']):
                    if results_dict['truth'][i] == clss:
                        if results_dict['truth'][i] == results_dict['prediction'][i]:
                            correct_class_cnt += 1
                        else:
                            incorrect_class_cnt += 1
                        total_cnt += 1
            self.metrics_dict['target'].append(clss)
            self.metrics_dict['correct'].append(correct_class_cnt)
            self.metrics_dict['incorrect'].append(incorrect_class_cnt)
            self.metrics_dict['total'].append(total_cnt)
        return self.metrics_dict

    def getRawMetrics(self):
        ##place holder for tp, fp, fn ...
        pass

    def calculatePrecision(self):
        ##place holder for calculating precision
        pass

    def calculateRecall(self):
        ##place holder for calculating recall
        pass

    def calculateF1(self):
        ##place holder for calculating F1 score
        pass

    ##This function calculate the accuracy of the model's output of the test set for classification
    def calculateAccuracy(self):
        classes = list(set(self.metrics_dict['target']))
        class_accuracy_list = []
        for clss in classes:
            clss_index = self.metrics_dict['target'].index(clss)
            correct_cnt = self.metrics_dict['correct'][clss_index]
            total_cnt = self.metrics_dict['total'][clss_index]
            class_accuracy = (correct_cnt / total_cnt)*100
            class_accuracy_list.append(class_accuracy)
            # print('class: ', clss, 'Accuracy(%): ', round(class_accuracy, 2))

            ##caclulate toatal accuracy
            avg_accuracy = np.mean(np.array(class_accuracy_list))
        return avg_accuracy

    ##This function calculates mse and is only used on regression
    def calculateMSE(self):
        print('Evaluating...')
        ##keep track of k value and mse
        mse_dict = {'k_value': [], 'mse': []}
        
        ##iterate through test sets
        cnt = 0
        
        for results_dict in self.results_list:
            # for k in results_dict['k_value']:
            error_list = [] ##record keeping for y - y_hat calculations
            for i in range(len(results_dict['prediction'])):
                ##find y - y_hat for each y in test set
                error = results_dict['truth'][i] - results_dict['prediction'][i]
                error_sqrd = error**2  ##error squared
                error_list.append(error_sqrd) ##record
            cnt += 1

            ##mse = 1/n(sum(y - y_hat)^2)
            mse = sum(error_list) / len(results_dict['prediction'])
            mse_dict['mse'].append(mse)
            mse_dict['k_value'].append(results_dict['k_value'])
            ##record values
            # print('average_mse: ', mse)

        ##take the average of each mse value per test set
        avg_mse = sum(mse_dict['mse']) / len(mse_dict['mse'])
        # print('average_mse: ', avg_mse)
        return avg_mse