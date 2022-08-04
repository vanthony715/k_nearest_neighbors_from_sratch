# -*- coding: utf-8 -*-
"""
@author: Anthony J. Vasquez
email: avasque1@jh.edu
phone: 315-920-8877

Class: Introduction to Machine Learning

Description: All Distance Methods Calculated Here
"""
import math
import numpy as np

class EuclideanDistance:
    
    '''Calculates the Gaussian Kernal Weight per Query'''
    
    def __init__(self, query_vector, vector):
        self.query_vector = query_vector
        self.vector = vector
        
    ##calculate the Euclidean distance
    def euclideanDistance(self):
        distance = math.sqrt(sum((element_1 - element_2)**2 for element_1,
                            element_2 in zip(self.query_vector, self.vector)))
        return distance