# -*- coding: utf-8 -*-
"""
YOSHIHIRO UEDA, MASAYUKI AKIYAMA, GÜNTHER HASINGER, TAKAMITSU MIYAJI,
MICHAEL G. WATSON, 2015, ???, ???, ???

This model is the more complicated version of the Ueda et al. (2003) 
double power law and piecewise redshift evolution in the QSO X-ray 
luminosity function.
"""

import numpy as np
from ueda2003 import _evolution_factor_pde, _evolution_factor_ldde, \
    _DoublePowerLaw

#-------------------------------------------------


qsolf_LDDE2_hardpars = \
{
 'A': 10**-6 *.70**3 * 2.91,
 'loglstar': 10**43.97,
 'gamma1': 0.96,
 'gamma2': 2.71,
 'p1': 4.78,
 'p2': -1.5,
 'p3': -6.2,
 'beta1': 0.84,
 'zstar': 1.86,
 'zstarc2': 3.0,
 'logLa1': 10**44.61,
 'logLa2': 10**45.67,#???
 'alpha1': 0.29,
 'alpha2': -0.1
}

qsolf_LDDE2_harderr = \
{
 'A_err': 0.07, 
 'loglstar_err': 10**0.06,
 'gamma1_err': 0.04,
 'gamma2_err': 0.09,
 'p1_err': 0.16,
 'p2_err': 0,
 'p3_err': 0,
 'beta1_err': 0.18,
 'zstar_err': 0.07,
 'zstarc2_err': 0,
 'logLa_err': 10**0.07,
 'logLa2_err': 0,
 'alpha_err': 0.02,
 'alpha2_err': 0
}

def _zc_of_L(L, **kwargs):
    """
    Compute cutoff redshift for luminosity-dependent density evolution.
    """
        
    La = 10**kwargs['logLa']

    if L < La:
        zc_ast = kwargs['zc'] * (L / La)**kwargs['alpha']
    elif L >= La:
        zc_ast = kwargs['zc']
        
    return zc_ast

def _evolution_factor(z, **kwargs):
    
    if z < kwargs['zc1']:
        eofz = (1. + z)**p1
    elif kwargs['zc1'] < z < kwargs['zc2']:
        eofz = (1. + kwargs['zc1'])**kwargs['p1'] \
            * ((1. + z) / (1. + kwargs['zc1']))**p2
    else:
        eofz = (1. + kwargs['zc1'])**kwargs['p1'] \
            * ((1. + kwargs['zc2']) / (1+kwargs['zc1']))**kwargs['p2'] \
            * ((1. + z) / (1. + kwargs['zc2']))**kwargs['p3']

    return eofz

def _evolution_factor_ldde(z, L, **kwargs):

    try:
        
        kw = kwargs.copy()
        for i in range(1, 2):
            kw['zc'] = kwargs['zc%i' % i]
            kwargs['zc%i' % i] = _zc_of_L(z, L, **kw)
        
        eofz = _evolution_factor(z, **kwargs)
    except ValueError:
        eofz = np.zeros_like(L)        
        zcarr = np.array(map(lambda LL: _zc_of_L(LL, **kwargs), L))
        for i, zcval in enumerate(zcarr):
            kwargs['zc'] = zcval
            eofz[i] = _evolution_factor_pde(z, **kwargs)
            
    return eofz

#-------------------------------------------------  

def randomsamples(samples, K = None, loglstar = None, \
gamma1 = None, gamma2 = None, p1 = None, p2  = None,\
p3  = None, beta1 = None, zstar = None, zstarc2  = None, 
logLa = None, logLa2 = None, alpha = None, alpha2 = None,\
K_err = None, loglstar_err = None, gamma1_err = None, 
gamma2_err = None, p1_err = None, p2_err = None, p3_err = None, \
beta1_err = None, zstar_err = None, zstarc2_err = None,\
logLa_err = None, logLa2_err = None, alpha_err = None, \
alpha2_err = None, **kwargs):

    randomsamples = []
    for i in range(samples):
        
        randomsample = {
        #'K': np.random.normal(K, K_err, samples),
        'A': 10**-6 *.70**3 * 2.91,
        'loglstar': np.random.normal(loglstar, loglstar_err, samples)[i],\
        'gamma1': np.random.normal(gamma1, gamma1_err, samples)[i],\
        'gamma2': np.random.normal(gamma2, gamma2_err, samples)[i],\
        'p1': np.random.normal(p1, p1_err, samples)[i],\
        'p2': -1.5,\
        'p3': -6.2,\
        'beta1': np.random.normal(beta1, beta1_err, samples)[i],\
        'zstar': np.random.normal(zstar, zstar_err, samples)[i],\
        'zstarc2': 3.0,\
        'logLa': np.random.normal(logLa, logLa_err, samples)[i],\
        'logLa2': 10**45.67,\
        'alpha': np.random.normal(alpha, alpha_err, samples)[i],\
        'alpha2': -0.1\
        }
        randomsamples.append(randomsample)
    return randomsamples
   

#-------------------------------------------------

def LuminosityFunction_LDDE(Lx, z, loglstar = None, A = None, gamma1 = None, gamma2 = None, p1 = None, p2  = None,\
p3  = None, beta1 = None, zstar = None, zstarc2  = None, logLa = None, logLa2 = None, alpha = None, alpha2 = None, **kwargs):

    
    if Lx <= logLa:
        zc1 = zstar*(Lx / logLa)**alpha
    elif Lx > logLa:
        zc1 = zstar
    
    if Lx <= logLa2:
        zc2 = zstarc2*(Lx / logLa2)**alpha2
    elif Lx > logLa2:
        zc2 = zstarc2
##################################################       
    if z <= zc1:
        ex = (1+z)**p1
    elif zc1 < z <= zc2:
        ex = (1+zc1)**p1*((1+z)/(1+zc1))**p2
    elif z > zc2:
        ex = (1+zc1)**p1*((1+zc2)/(1+zc1))**p2*((1+z)/(1+zc2))**p3
   
    return  A * ((Lx / loglstar)**gamma1 + (Lx / loglstar)**gamma2)**-1 * ex
    
    
    
    