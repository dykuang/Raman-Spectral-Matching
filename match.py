# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 16:05:16 2018

@author: dykua

This script does the actual matching
"""
import numpy as np
from DataExtraction import extract_data
from Util import preprocess, smooth1D, find_top_peak_loc, multi_gaussian

lib_data = extract_data('../Li-Lin - data/Spectral Data')
test_data = extract_data('../Li-Lin - data/Test Data/Caffeine Labspec Run Used')
#test_data = extract_data('../Li-Lin - data/Test Data/Ephedrine HCL Labspec Run Used')
#test_data = extract_data('../Li-Lin - data/Test Data/Acetaminophen Labspec Run Used')
#test_data = extract_data('../Li-Lin - data/Test Data/Oxycocet Labspec Run Used')
#test_data = extract_data('../Li-Lin - data/Test Data/Benzene derivatives (baseline corrected)')
#test_data = extract_data('../Li-Lin - data/Test Data/Benzene Chloroform')


lb = 200 
up = 1800
resolution = 1
sd_prior = 5
filter_width = np.arange(1,20)
peak_num = 6

#------------------------------------------------------------------------------
# Preprocess
#------------------------------------------------------------------------------
    
test_data = preprocess(test_data, lb, up, resolution)
lib_data = preprocess(lib_data, lb, up, resolution)

name_in_test = list(test_data.keys())
name_in_lib = list(lib_data.keys())

test_ind = 0
test_sample = test_data[name_in_test[test_ind]]
print('test_samle name: {}'.format(name_in_test[test_ind]))

#------------------------------------------------------------------------------
# Get signature from test_sample
#------------------------------------------------------------------------------
test_sample[1,:] = smooth1D(test_sample[1,:])
loc, height = find_top_peak_loc(test_sample, peak_num, filter_width)
sd = sd_prior*np.ones_like(loc)
mg = multi_gaussian(test_sample[0,:], loc, height, sd, use_height = False)

#------------------------------------------------------------------------------
# Get signature also for lib data? 
#------------------------------------------------------------------------------
#for key, value in lib_data.items():
#    value[1,:] = smooth1D(value[1,:])
#    loc, height = find_top_peak_loc(value, peak_num, filter_width)
#    lib_data[key]= np.vstack([value[0,:], multi_gaussian(value[0,:], loc, height,
#                            sd, use_height = False)])

#------------------------------------------------------------------------------
# Calculate matching score
#------------------------------------------------------------------------------
match_score = np.zeros(len(name_in_lib))
from metrics import Dot
for i, name in enumerate(name_in_lib):
    match_score[i] = Dot(mg, lib_data[name][1,:])

candidates_order = np.argsort(match_score)[::-1]
for i, c in enumerate(candidates_order[:10]):
    print("The top {0} selection are {1} with score: {2:4f}.".
          format(i+1, name_in_lib[c], match_score[c]))

#==============================================================================
# The above match works pretty good and fast, tries the histogram method below
# The result is not very good...
#==============================================================================
#from Util import to_hist
#peak_num_hist = 120
#bins = 40
#window_len = 11
#half_window_len = int(window_len/2)
#
#for key, value in test_data.items():
#        mask = (value[0,:]>lb) & (value[0,:] < up)
#        test_data[key] = np.stack((value[0, mask], smooth1D(value[1,mask]/np.linalg.norm(value[1, mask]), window_len)))
#
#for key, value in lib_data.items():
#        mask = (value[0,:]>lb) & (value[0,:] < up)
#        lib_data[key] = np.stack((value[0, mask], smooth1D(value[1,mask]/np.linalg.norm(value[1, mask]), window_len)))
#
#test_ind = 6
#test_sample = test_data[name_in_test[test_ind]]
#print('test_samle name: {}'.format(name_in_test[test_ind]))
#test_hist = to_hist(test_sample, peak_num_hist, filter_width, bins)
#
#for key, value in lib_data.items():
##    value[1,:] = smooth1D(value[1,:])
#    lib_data[key]= to_hist(value, peak_num_hist, filter_width, bins)
#
#match_score = np.zeros(len(name_in_lib))
#from metrics import JSD
#for i, name in enumerate(name_in_lib):
#    match_score[i] = JSD(test_hist, lib_data[name])
#
#candidates_order = np.argsort(match_score)
#for i, c in enumerate(candidates_order[:]):
#    print("The top {0} selection are {1} with distance: {2:4f}.".
#          format(i+1, name_in_lib[c], match_score[c]))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
