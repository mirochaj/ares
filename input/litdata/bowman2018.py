"""

bowman2018.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Fri Apr 27 16:06:50 PDT 2018

Description: 

"""

import os
import numpy as np
import matplotlib.pyplot as pl
from ares.util.Stats import get_nu

_input = os.getenv('ARES') + '/input/edges/'

def _load(fig=2):
    """
    Returns (frequencies, temperatures) for Fig. 1 or 2 data.
    
    ..note:: The contents of the `data` array are different for each file!
    
    """
    raw = np.loadtxt('{}/figure{}_plotdata.csv'.format(_input, fig), 
        unpack=True, delimiter=',', skiprows=1)
    
    if fig == 1:
        freq, weight = raw[0:2]
        
        data = raw[2:] * 1e3
        mask = np.logical_not(weight)
        Tarr = np.ma.array(data, mask=np.array([mask]*len(data)))
    else:
        freq, z, age = raw[0:3]
        
        data = raw[4::2] * 1e3
        mask = np.logical_not(raw[3::2])
        Tarr = np.ma.array(data, mask=mask)

    return freq, Tarr
    
def gauss_flat(freq, A=-500., freq0=78., fwhm=18.7, tau=7.):
    """
    Flattened Gaussian model for the brightness temperature.
    """
    B = (4. * (freq - freq0)**2 / fwhm**2) \
      * np.log(-np.log((1. + np.exp(-tau)) / 2.) / tau)

    return A * ((1. - np.exp(-tau * np.exp(B))) / (1. - np.exp(-tau)))

def generate_bands(N=1000, seed=None):
    
    freq, Tarr = _load()
    
    A = -500.
    tau = 7.
    freq0 = 78.
    fwhm = 19.
    freq0_err = (1., 1.) # MHz
    tau_err = (5., 3.)
    fwhm_err = (4., 2.)
    A_err = (200, 200)#(200., 500.)

    pars = ['A', 'freq0', 'fwhm', 'tau']
    vals = [A, freq0, fwhm, tau]
    errs = [A_err, freq0_err, fwhm_err, tau_err]

    # Convert from 0.99 confidence intervals to 1-sigma

    np.random.seed(seed)
        
    samples = []
    for k, par in enumerate(pars):
        err = np.mean(errs[k])
        s1_err = get_nu(err, 0.99, 0.68)
        samples.append(np.random.normal(vals[k], s1_err, N))

    samples = np.array(samples)

    size = len(freq)
    freq_rep = np.repeat(freq, N).reshape(N, size)

    y = np.vstack([gauss_flat(freq, *samples.T[i]) for i in range(N)])

    s1 = np.percentile(y, (16, 84), axis=0)
    s2 = np.percentile(y, (2.5, 97.5), axis=0)
    
    return freq, s1, s2

def plot_bands(ax=None, N=1000):
    if ax is None:
        fig = pl.figure(111); ax = fig.add_subplot(111)

    zax = None
    
    freq, s1, s2 = generate_bands(N)
    
    ax.fill_between(freq, *s2, color='k', alpha=0.1)
    ax.fill_between(freq, *s1, color='k',alpha=0.3)  

    return ax, zax
    
def plot_recovered(ax=None, **kwargs):
    
    freq, Tarr = _load()
    
    if ax is None:
        fig = pl.figure(111); ax = fig.add_subplot(111)

    zax = None
    
    if kwargs == {}:
        ls = ['-'] * 6 + ['-.']
        colors = 'r', 'k', 'b', 'g', 'orange', 'yellow', 'brown'
        zorder = [1, 10, 1, 1, 1, 1, 1]
        alpha = [0.5, 1, 0.5, 0.5, 0.5, 0.5, 0.5]
        
        for i, spectrum  in enumerate(Tarr):
            ax.plot(freq, spectrum, color=colors[i], ls=ls[i], alpha=alpha[i], 
                zorder=zorder[i])
    else:
        for i, spectrum  in enumerate(Tarr):
            ax.plot(freq, spectrum, **kwargs)            

    return ax, zax
