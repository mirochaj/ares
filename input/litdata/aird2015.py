# -*- coding: utf-8 -*-
import numpy as np

"""
J. Aird1, A. L. Coil, A. Georgakakis, K. Nandra, G. Barro, P. G. P´erez-Gonz´alez, 2015, ???, ???, ???

There are two different models with two different sets of Data each:

Model 1: LDDE1
    -Simple model of Ueda 2003 (I Believe*** will check and fix**)
    
Model 2: LDDE2:
    -Complex model of UEDA 2014 (I  Believe** will check and fix**)
    
Data set 1:
    -Soft band (0.5-2 KeV) parameters for both models 
    
Data set 2:
    -Hard band (2-7 KeV) parameters for both models 
    
NOTE: 
    
    These models produce Lx(z). This does *NOT* integrate and give the Lx density for either model.
    
    --- = spacing between different sections of code
    
    ### = spacing within a section of code to denote a different section within that particular code
    
    $$$ = spacing between different parts of codes

"""

#-------------------------------------------------

qsolf_LDDE2_softpars = \
{
 'K': 10**-5.97,#???
 'loglstar': 10**44.18,
 'gamma1': 0.79,
 'gamma2': 2.55,
 'p1': 4.35,
 'p2': -0.96,
 'p3': -7.84,
 'beta1': 0.65,
 'zstar': 1.80,
 'zstarc2': 3.09,
 'logLa': 10**44.92,
 'logLa2': 10**44.27,
 'alpha': 0.16,
 'alpha2': 0.06
}

qsolf_LDDE2_softerr = \
{
 'K_err': 10**0.05,#???
 'loglstar_err': 10**0.03,
 'gamma1_err': 0.01,
 'gamma2_err': 0.05,
 'p1_err': 0.35,
 'p2_err': 0.12,
 'p3_err': 0.51,
 'beta1_err': 0.06,
 'zstar_err': 0.08,
 'zstarc2_err': 0.11,
 'logLa_err': 10**0.12,
 'logLa2_err': 10**0.30,
 'alpha_err': 0.01,
 'alpha2_err': 0.02
}

qsolf_LDDE1_softpars = \
{
 'K': 10**-5.87,#???
 'loglstar': 10**44.17,
 'gamma1': 0.67,
 'gamma2': 2.37,
 'p1': 3.67,
 'p2': -2.92,
 'zstar': 2.27,
 'logLa': 10**0.92,
 'alpha': 0.18,
}

qsolf_LDDE1_softerr = \
{
 'K_err': 10**0.05,#???
 'loglstar_err': 10**0.04,
 'gamma1_err': 0.02,
 'gamma2_err': 0.06,
 'p1_err': 0.09,
 'p2_err': 0.14,
 'zstar_err': 0.09,
 'logLa_err': 10**0.11,
 'alpha_err': 0.01,
}
##################################################
qsolf_LDDE2_hardpars = \
{
 'K': 10**-5.72,#???
 'loglstar': 10**44.09,
 'gamma1': 0.73,
 'gamma2': 2.22,
 'p1': 4.34,
 'p2': -0.30,
 'p3': -7.33,
 'beta1': -0.19,
 'zstar': 1.85,
 'zstarc2': 3.16,
 'logLa': 10**44.78,
 'logLa2': 10**44.46,
 'alpha': 0.23,
 'alpha2': 0.12
}

qsolf_LDDE2_harderr = \
{
 'K_err': 10**0.07,#???
 'loglstar_err': 10**0.05,
 'gamma1_err': 0.02,
 'gamma2_err': 0.06,
 'p1_err': 0.18,
 'p2_err': 0.13,
 'p3_err': 0.62,
 'beta1_err': 0.09,
 'zstar_err': 0.08,
 'zstarc2_err': 0.10,
 'logLa_err': 10**0.07,
 'logLa2_err': 10**0.17,
 'alpha_err': 0.01,
 'alpha2_err': 0.02
}

qsolf_LDDE1_hardpars = \
{
 'K': 10**-5.63,#???
 'loglstar': 10**44.10,
 'gamma1': 0.72,
 'gamma2': 2.26,
 'p1': 3.97,
 'p2': -2.08,
 'zstar': 2.02,
 'logLa': 10**44.71,
 'alpha': 0.20,
}

qsolf_LDDE1_harderr = \
{
 'K_err': 10**0.07,#???
 'loglstar_err': 10**0.05,
 'gamma1_err': 0.02,
 'gamma2_err': 0.07,
 'p1_err': 0.17,
 'p2_err': 0.17,
 'zstar_err': 0.09,
 'logLa_err': 10**0.09,
 'alpha_err': 0.01,
 }
 
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
        'K': 10**-5.72,
        'loglstar': np.random.normal(loglstar, loglstar_err, samples)[i],\
        'gamma1': np.random.normal(gamma1, gamma1_err, samples)[i],\
        'gamma2': np.random.normal(gamma2, gamma2_err, samples)[i],\
        'p1': np.random.normal(p1, p1_err, samples)[i],\
        'p2': np.random.normal(p2, p2_err, samples)[i],\
        'p3': np.random.normal(p3, p3_err, samples)[i],\
        'beta1': np.random.normal(beta1, beta1_err, samples)[i],\
        'zstar': np.random.normal(zstar, zstar_err, samples)[i],\
        'zstarc2': np.random.normal(zstarc2, zstarc2_err, samples)[i],\
        'logLa': np.random.normal(logLa, logLa_err, samples)[i],\
        'logLa2': np.random.normal(logLa2, logLa2_err, samples)[i],\
        'alpha': np.random.normal(alpha, alpha_err, samples)[i],\
        'alpha2': np.random.normal(alpha2, alpha2_err, samples)[i],\
        }
        randomsamples.append(randomsample)
    return randomsamples
  
#-------------------------------------------------    

def LuminosityFunction(Lx, z, LDDE1 = None, LDDE2 = None, K= None, loglstar = None, gamma1 = None, gamma2 = None, p1 = None, \
p2 = None, p3 = None, beta1 = None, zstar = None, zstarc2 = None, logLa = None, logLa2 = None, \
alpha = None, alpha2 = None):

    """This function is the Luminosity Density Dependent Equation (LDDE). There are two different models to choose from, LDDE1 and LDDE2.
    LDDE2 is more complicated but seems to be the model of choice for authors. ***More to come""" 

    if LDDE1 == True:
        
        if Lx < logLa:
            zc1 = zstar*(Lx / logLa)**alpha
        elif Lx >= logLa:
            zc1 = zstar
    
##################################################   

        if z <= zc1:
            ex = (1+z)**p1
        elif z > zc1:
            ex = (1+zc1)**p1*((1+z)/(1+zc1))**p2
           
        return  K * ((Lx / loglstar)**gamma1 + (Lx / loglstar)**gamma2)**-1 * ex
        
        
    elif LDDE2 == True:    
        
        if Lx < logLa:
            zc1 = zstar*(Lx / logLa)**alpha
        elif Lx >= logLa:
            zc1 = zstar
    
##################################################   
    
        if Lx < logLa2:
            zc2 = zstarc2*(Lx / logLa2)**alpha2
        elif Lx >= logLa2:
            zc2 = zstarc2 
    
##################################################  
      
        if z < zc1:
            ex = (1+z)**p1
        elif zc1 < z < zc2:
            ex = (1+zc1)**p1*((1+z)/(1+zc1))**p2
        elif z > zc2:
            ex = (1+zc1)**p1*((1+zc2)/(1+zc1))**p2*((1+z)/(1+zc2))**p3


        return  K * ((Lx / loglstar)**gamma1 + (Lx / loglstar)**gamma2)**-1 * ex
        
    else:
        
        raise TypeError('Pick a Luminosity Dependent Density Evolution')

#-------------------------------------------------  