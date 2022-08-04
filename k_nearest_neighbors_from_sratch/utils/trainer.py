#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Anthony J. Vasquez
email: avasque1@jh.edu
phone: 315-920-8877

Class: Introduction to Machine Learning

Description: Functions that takes dataset and algorthm as input and trains
"""

class TrainModel:

    '''Trainer Model'''

    def __init__(self, model, X_train, y_train, savepath):
        print('Trainer Initialized')
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.savepath = savepath

    def train(self):
        print('Training ...')
        learner_output = self.model.model()
        return learner_output

    def saveWeights(self):
        print('Saving Weights to: ', self.savepath)



