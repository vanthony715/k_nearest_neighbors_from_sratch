# -*- coding: utf-8 -*-
"""
@author: Anthony J. Vasquez
email: avasque1@jh.edu
phone: 315-920-8877

Class: Introduction to Machine Learning

Description: All Kernal Methods Calculated Here
"""
import math
import numpy as np

class GaussianKernal:
    
    '''Calculates the Gaussian Kernal Weight per Query'''
    
    def __init__(self, query_vector, vector, sigma):
        self.query_vector = query_vector
        self.vector = vector
        self.sigma = sigma

    def gaussianKernal(self):
        gaussian_kernal = math.exp(-np.linalg.norm(self.query_vector - self.vector, 
                                                   2)**2/(2.*self.sigma**2))
        return gaussian_kernal