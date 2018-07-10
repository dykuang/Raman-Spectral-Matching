# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 09:52:11 2018

@author: dykua

Metrics used for matching

"""
import numpy as np
from scipy.stats import entropy, wasserstein_distance, energy_distance

def ED(x, y):
    '''
    Euclidian distance. x, y are locations of top peaks, y is the reference
    '''
    return (1+np.linalg.norm(x-y)/np.linalg.norm(y))**(-1)

def CC(x, y):
    '''
    cross-corelation, x, y are two vectors
    '''
    return np.corrcoef(x, y)[0,1]

def Dot(x, y):
    return np.sum(x*y)/np.linalg.norm(x)/np.linalg.norm(y)

def top_peak_dist(x, y, sort = False):
    
    if sort:
        x_sorted = np.sort(x)
        y_sorted = np.sort(y)
    else:
        x_sorted = x
        y_sorted = y
        
    return ED(x_sorted, y_sorted)

def JSD(P, Q):
    '''
    Jensen- Shannon distance
    '''
    _P = P / np.linalg.norm(P, ord=1)
    _Q = Q / np.linalg.norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return (0.5 * (entropy(_P, _M) + entropy(_Q, _M)))
