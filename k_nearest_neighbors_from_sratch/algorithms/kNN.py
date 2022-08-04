#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Anthony J. Vasquez
email: avasque1@jh.edu
phone: 315-920-8877

Class: Introduction to Machine Learning

Description: K-Nearest-Neighbors Algorithm
"""
import numpy as np
import pandas as pd
from math import sqrt

from algorithms.kernals import GaussianKernal
from algorithms.distances import EuclideanDistance


class KNN:
    '''
    calculates the K-Nearest Neighbors
    '''

    def __init__(self, X_train, test_row, k_neighbors, editted_knn, condensed_knn,
                 mode, max_error, sigma):
        self.X_train = X_train  # train data
        self.test_row = test_row  # row to test against
        self.k_neighbors = k_neighbors  # how many nearest neighbors
        self.editted_knn = editted_knn  # should knn be supplemented by editting
        self.condensed_knn = condensed_knn  # should knn be condensed
        self.mode = mode  # either classification or regression depending on dataset
        self.max_error = max_error
        self.sigma = sigma  # value for gaussian kernal weighting

    def distanceCalc(self):

        # track distance and class
        self.dist_dict = {'index': [], 'distance': [], 'target': [], 
                          'gauss_weights': [], 'test_row_class': []}
        
        # delete target value of row for calculation for norm calculation
        test_row = np.array(self.test_row)
        test_row = np.delete(test_row, -1)
        
        cnt = 0
        # use a test row (query) and iterate through the rest of the rows
        for index, row in self.X_train.iterrows():

            # delete target value of row for calculation for norm calculation
            row = np.array(row)
            row = np.delete(row, -1)

            # take the square root and square to obtain distances relative to the test_vector
            distance = EuclideanDistance(test_row, row).euclideanDistance()
            
            ##use gaussian kernal as a second test
            gauss_weights = GaussianKernal(test_row, row, self.sigma).gaussianKernal()

            # append to distance dict
            self.dist_dict['distance'].append(distance)
            self.dist_dict['target'].append(self.X_train['target'][index])
            self.dist_dict['index'].append(cnt)
            self.dist_dict['gauss_weights'].append(gauss_weights)
            self.dist_dict['test_row_class'].append(np.array(self.test_row)[-1])
            cnt += 1

    def getNearestNeighbors(self):
        # get column names
        col_names = list(self.dist_dict.keys())
        # create distance dataframe
        dist_df = pd.DataFrame(self.dist_dict, columns=col_names)

        # sort by distance to insure values are in ascending order
        dist_df = dist_df.sort_values('distance')

        # keep track of nearest neighbors
        self.nn_dict = {'nearest_neighbor': [],'nn_class': [], 'index': [], 
                        'gauss_weights': [], 'test_row_class': []}

        # for a preselected k_neighbor range, iterate throufh values and check predictions
        for k in range(self.k_neighbors):
            # get index assuming list in ascending order
            index = int(dist_df['index'].iloc[[k]])

            # get nearest neighbor
            neighbor = self.X_train.values[index]
            neighbor_class = neighbor[-1]

            # take target out of neighbor
            neighbor = np.delete(neighbor, -1)
            
            ##guassian weights
            gauss_weights = np.array(dist_df['gauss_weights'].iloc[[k]])
            
            # append to the nearest neighbor list
            self.nn_dict['nearest_neighbor'].append(neighbor)
            self.nn_dict['nn_class'].append(neighbor_class)
            self.nn_dict['index'].append(index)
            self.nn_dict['gauss_weights'].append(gauss_weights)
            self.nn_dict['test_row_class'].append(dist_df['test_row_class'].iloc[[k]])

        return self.nn_dict

    def regress(self):
        # calculate the closest number to mean for regression
        indices_to_drop = []  # ghost list used for classification

        # keep track of rows without 'target'
        nn_list = []
        
        ##drop the indices that are the min guassian value for editted knn
        indices_to_drop = []

        # drop target for each nearest neighbor and append to list
        min_gauss = min(self.nn_dict['gauss_weights'])
        for i in range(len(self.nn_dict['nn_class'])):
            if min_gauss[0] < self.nn_dict['gauss_weights'][i][0]:
                nn_list.append(self.nn_dict['nn_class'][i])
            else:
                
                index = self.nn_dict['index'][i]
                indices_to_drop.append(index)
                
        # get the nearest neighbor mean value
        predicted_value = round(np.mean(nn_list), 0)
        return predicted_value, indices_to_drop

    def predictClass(self):
        # if self.mode == 'classification':
        # keep track nearest neighbors
        unique_nn_dict = {'target': [], 'count': []}

        # keep track of class for plurality test
        unique_nearest_neighbors = list(set(self.nn_dict['nn_class']))

        # for each of the unique nearest neighbors
        for neighbor in unique_nearest_neighbors:
            cnt = 1
            # count how many instances of each class there are
            for nn_list_neighbor in self.nn_dict['nn_class']:
                if nn_list_neighbor == neighbor:
                    unique_nn_dict['target'].append(neighbor)
                    unique_nn_dict['count'].append(cnt)
                    cnt += 1

        # find the class with the max count to complete plurality test
        max_count = max(unique_nn_dict['count'])
        index = unique_nn_dict['count'].index(max_count)
        max_class = unique_nn_dict['target'][index]

        if self.editted_knn or self.condensed_knn:
            indices_to_drop = []
            for clss in self.nn_dict['nn_class']:
                if max_class != clss:
                    index = self.nn_dict['nn_class'].index(clss)
                    indices_to_drop.append(self.nn_dict['index'][index])
            return max_class, indices_to_drop

        else:
            return max_class, index
