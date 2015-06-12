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
 'K': 10**0.05,#???
 'loglstar': 10**0.03,
 'gamma1': 0.01,
 'gamma2': 0.05,
 'p1': 0.35,
 'p2': -0.12,
 'p3': -0.51,
 'beta1': 0.06,
 'zstar': 0.08,
 'zstarc2': 0.11,
 'logLa': 10**0.12,
 'logLa2': 10**0.30,
 'alpha': 0.01,
 'alpha2': 0.02
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
 'K': 10**0.05,#???
 'loglstar': 10**0.04,
 'gamma1': 0.02,
 'gamma2': 0.06,
 'p1': 0.09,
 'p2': 0.14,
 'zstar': 0.09,
 'logLa': 10**0.11,
 'alpha': 0.01,
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
 'K': 10**0.07,#???
 'loglstar': 10**0.05,
 'gamma1': 0.02,
 'gamma2': 0.06,
 'p1': 0.18,
 'p2': 0.13,
 'p3': 0.62,
 'beta1': 0.09,
 'zstar': 0.08,
 'zstarc2': 0.10,
 'logLa': 10**0.07,
 'logLa2': 10**0.17,
 'alpha': 0.01,
 'alpha2': 0.02
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
 'K': 10**0.07,#???
 'loglstar': 10**0.05,
 'gamma1': 0.02,
 'gamma2': 0.07,
 'p1': 0.17,
 'p2': 0.17,
 'zstar': 0.09,
 'logLa': 10**0.09,
 'alpha': 0.01,
 }
 
#-------------------------------------------------

def _LuminosityFunction_LDDE1(Lx, z, K = None, loglstar=None, A=None, gamma1=None, gamma2=None, p1=None, \
p2 = None, beta1=None, zstar=None, logLa=None, alpha=None, **kwargs):

    """This equation is from Ueda et al. (2003) and is used by Aird et al. (2015)"""

        
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

#-------------------------------------------------

def _LuminosityFunction_LDDE2(Lx, z, K= None, loglstar = None, gamma1 = None, gamma2 = None, p1 = None, \
p2 = None, p3 = None, beta1 = None, zstar = None, zstarc2 = None, logLa = None, logLa2 = None, \
alpha = None, alpha2 = None):
    
    """This equation is taken from Aird et al. (2015) and is based off of Ueda et al. (2014) LDDE equation. 
    This equation stands from the Luminosity Dependent Desnity Equation (LDDE) used to describe the 
    evolutoion of X-ray luminosity from AGNs on the basis of X-ray surveys. The X-ray luminosity Function 
    (XLF) is best described using the LDDE model. The LDDE Model is a function of redshift and X-ray 
    Luminosity best fitting within the range of 0-5 and 10**42-10**48, respectively. This model is the 
    modified version of Ueda et al. (2003) LDDE model. This model accounts for the decay in the comoving 
    numebr density of luminous AGN.""" 
    
    #e1 = p1 + beta1*(Lx - 10**44.48) 
    #I need to include this for the LDDE2 model but even when I add
    #a simple version the code crashes... Jordan? This can be found in Arid et al. (2015), pg. 14
        
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

    p = (K * ((Lx / loglstar)**gamma1 + (Lx / loglstar)**gamma2)**-1 * ex)
    print p
    return  K * ((Lx / loglstar)**gamma1 + (Lx / loglstar)**gamma2)**-1 * ex
    
#-------------------------------------------------    
 
def LuminosityFunction_LDDE1_hardband(Lx, z, **kwargs):

    return _LuminosityFunction_LDDE1(Lx, z, **qsolf_LDDE1_hardpars)
    
def LuminosityFunction_LDDE2_hardband(Lx, z, **kwargs):

    return _LuminosityFunction_LDDE2(Lx, z, **qsolf_LDDE2_hardpars)
    
def LuminosityFunction_LDDE1_softband(Lx, z, **kwargs):

    return _LuminosityFunction_LDDE1(Lx, z, **qsolf_LDDE1_softpars)
    
def LuminosityFunction_LDDE2_softband(Lx, z, **kwargs):

    return _LuminosityFunction_LDDE2(Lx, z, **qsolf_LDDE2_softpars)
    
