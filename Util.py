# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 09:51:30 2018

@author: dykua

Some utility functions for matching algorithms
"""
import numpy as np
from scipy.signal import find_peaks_cwt
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def square_rescale(spectrum):
    [X, Y] = spectrum
    maxY = max(Y)
    Y /= maxY
    Y = [y**2 for y in Y]
    return np.array([X, Y])

def check_signal(signals):
    plt.figure()
    plt.plot(signals[0,:], signals[1,:])

def find_top_peak_loc(X, num, width):
    '''
    find peaks and sorted by their height
    '''
    peak_ind = find_peaks_cwt(X[1,:], width)
    peaks = X[:,peak_ind]
    sort_ind = np.argsort(peaks[1,:])
    peaks_sorted_by_height = peaks[:, sort_ind]
    top_peak_loc = peaks_sorted_by_height[0,-num:]
    top_peak_height = peaks_sorted_by_height[1,-num:]
    
    return top_peak_loc[::-1], top_peak_height[::-1]

def to_hist(signal, peak_num, filter_width, bin_num):
    '''
    Change the spectra into "histogram of peak locations"
    '''
    l, h = find_top_peak_loc(signal,peak_num,filter_width)
    return np.histogram(l, bin_num, weights = h)[0]

def smooth1D(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise(ValueError, "Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    half_window_len = int(window_len/2)
    return y[half_window_len:-half_window_len]

def preprocess(data, lb, up, res = 1):
    '''
    truncate, smooth and normalize the signal in place.
    '''
    x_domain = np.arange(lb, up, res)
    for key, value in data.items():
            mask = (value[0,:]>lb) & (value[0,:] < up)
    #        data[key] = np.stack((value[0, mask], smooth1D(value[1,mask]/np.linalg.norm(value[1, mask]), window_len)[half_window_len:-half_window_len]))
#            data[key] = np.stack((value[0, mask], value[1,mask]/np.linalg.norm(value[1, mask])))
            data[key] = np.stack((value[0, mask], value[1,mask]/np.max(value[1, mask])))
    #        data[key] = square_rescale(value[:,mask])
            f = interp1d(value[0,mask], data[key], 'linear', fill_value= 'extrapolate')
            data[key] = f(x_domain)
            
    return data

def multi_gaussian(x, peak_loc, peak_height, 
                   std , use_height = False):
    '''
    Fit peak locations to a sum of gaussian
    '''
    g_num = len(peak_loc)
    vec_g = np.zeros([g_num, len(x)])
    for i in range(g_num):
        vec_g[i,:] = np.exp( -0.5*(x - peak_loc[i])**2/std[i]**2 )
#        print(max(vec_g[i,:]))
    if use_height:
        return np.sum(np.dot(np.diag(peak_height), vec_g),0)
    else:
        return np.sum(vec_g, 0)
#        
        
def MG_for_fit(x, *args):
    '''
    Function that passes to curve_fit
    args contains height|mean| sd |
    '''
    g_num = len(args)//3
    mix = 0
    for i in range(g_num):
        mix += args[i]*np.exp( -0.5*(x - args[g_num+i])**2/args[2*g_num+i]**2)
    
    return mix
    

    
if __name__ == '__main__':
    from DataExtraction import extract_data
    data = extract_data('Benzene Chloroform')
    lb = 200
    up = 1800
    resolution = 1
    top_peaks_num_lb = 6
    top_peaks_num_ub = 12
    filter_width = np.arange(1,20)
    peak_num = 4
    
    data = preprocess(data, lb, up, resolution)
    test_sample = data['benzene_pt5sec_01.txt']
    test_sample[1,:] = smooth1D(test_sample[1,:])
    loc, height = find_top_peak_loc(test_sample, peak_num, filter_width)
    sd = 5*np.ones_like(loc)
    mg = multi_gaussian(test_sample[0,:], loc, height, sd, use_height = True)
    from scipy.optimize import curve_fit
    ini = np.hstack([height, loc, sd])
#    ini = np.hstack([np.ones_like(loc), loc, sd])
    coeff, _ = curve_fit(MG_for_fit, test_sample[0,:], test_sample[1,:], p0=ini) # not stable if peak_num is large.
    mg_fitted = multi_gaussian(test_sample[0,:], coeff[peak_num:2*peak_num], 
                               coeff[:peak_num], coeff[2*peak_num:3*peak_num], 
                               use_height = True)
    plt.figure()
    plt.plot(test_sample[0,:], test_sample[1,:]) # original data
    plt.plot(test_sample[0,:], mg)  # fitted-gaussian using fixed std
    plt.plot(test_sample[0,:], mg_fitted) # fine-tuned fitted-gaussian
    plt.scatter(loc, height, marker = '^')

        
    