# -*- coding: utf-8 -*-
import numpy as np

"""
YOSHIHIRO UEDA, MASAYUKI AKIYAMA, GÜNTHER HASINGER, TAKAMITSU MIYAJI,
MICHAEL G. WATSON, 2015, ???, ???, ???

This model is the more complicated version of Ueda et al. (2003) double power law
  
  
NOTE: 
    
    --- = spacing between different sections of code
    
    ### = spacing within a section of code to denote a different section within that particular code

"""

#-------------------------------------------------


qsolf_LDDE_hardpars = \
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
 'logLa': 10**44.61,
 'logLa2': 10**45.67,#???
 'alpha': 0.29,
 'alpha2': -0.1
}

qsolf_LDDE_harderr = \
{
 'A': 0.07, 
 'loglstar': 10**0.06,
 'gamma1': 0.04,
 'gamma2': 0.09,
 'p1': 0.16,
 'p2': 0,
 'p3': 0,
 'beta1': 0.18,
 'zstar': 0.07,
 'zstarc2': 0,
 'logLa': 10**0.07,
 'logLa2': 0,
 'alpha': 0.02,
 'alpha2': 0
}

#-------------------------------------------------
def _LuminosityFunction_LDDE(Lx, z, loglstar = None, A = None, gamma1 = None, gamma2 = None, p1 = None, p2  = None,\
p3  = None, beta1 = None, zstar = None, zstarc2  = None, logLa = None, logLa2 = None, alpha = None, alpha2 = None):

    
    if Lx <= logLa:
        zc1 = zstar*(Lx / logLa)**alpha
    elif Lx > logLa:
        zc1 = zstar
    
    if Lx <= logLa2:
        zc2 = zstarc2*(Lx / logLa2)**alpha2
    elif Lx > logLa2:
        zc2 = zstarc2
#------------------------------------------------        
    if z <= zc1:
        ex = (1+z)**p1
    elif zc1 < z <= zc2:
        ex = (1+zc1)**p1*((1+z)/(1+zc1))**p2
    elif z > zc2:
        ex = (1+zc1)**p1*((1+zc2)/(1+zc1))**p2*((1+z)/(1+zc2))**p3
   
    return  A * ((Lx / loglstar)**gamma1 + (Lx / loglstar)**gamma2)**-1 * ex
 
def LuminosityFunction_LDDE(Lx, z, **kwargs):

    return _LuminosityFunction_LDDE(Lx, z, **qsolf_LDDE_hardpars)