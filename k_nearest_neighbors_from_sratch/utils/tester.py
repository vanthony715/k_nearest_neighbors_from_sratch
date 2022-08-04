#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Anthony J. Vasquez
email: avasque1@jh.edu
phone: 315-920-8877

Class: Introduction to Machine Learning

Description: Evaluate trained model
"""

class Tester:
    '''
    Test Trained Model
    '''
    def __init__(self, weights_path, trained_model_output, X_test, y_test):
        print('Tester Initialized')
        self.weights_path = weights_path
        self.trained_model_output = trained_model_output ##the real model will not use this input
        self.X_test = X_test
        self.y_test = y_test

    def loadWeights(self):
        print('Loading Weights from: ', self.weights_path)
        '''
        place holder for loading weights
        '''

    def test(self):
        print('Testing...')

        '''
        Place holder for testing
        '''
        return self.trained_model_output