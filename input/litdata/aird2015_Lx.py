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
    
    These models produce Lx(z) and does integrate and give the Lx density for either model.
    
    --- = spacing between different sections of code
    
    ### = spacing within a section of code to denote a different section within that particular code

    I have the errors included in each dictionary. 
"""

#-------------------------------------------------

qsolf_LDDE2_softpars_integration = \
{
 #'K': np.random.normal(10**-5.97, 10**0.05, 1000),
 'K': 10**-5.97,
 'loglstar': np.random.normal(10**44.18, 10**0.03, 1000),
 'gamma1': np.random.normal(0.79, 0.01, 1000),
 'gamma2': np.random.normal(2.55, 0.05, 1000),
 'p1': np.random.normal(4.35, 0.35, 1000),
 'p2': np.random.normal(-0.96, 0.12, 1000),
 'p3': np.random.normal(-7.84, 0.51, 1000),
 'beta1': np.random.normal(0.65, 0.06, 1000),
 'zstar': np.random.normal(1.80, 0.08, 1000),
 'zstarc2': np.random.normal(3.09, 0.11,1000),
 'logLa': np.random.normal(10**44.92, 10**0.12, 1000),
 'logLa2': np.random.normal(10**44.27, 10**0.30, 1000),
 'alpha': np.random.normal(0.16, 0.01, 1000),
 'alpha2': np.random.normal(0.06, 0.02, 1000)
}

qsolf_LDDE1_softpars_integration = \
{
 #'K': np.random.normal(10**-5.87, 10**0.05, 1000),
 'K': 10**-5.87,
 'loglstar': np.random.normal(10**44.17, 10**0.04, 1000),
 'gamma1': np.random.normal(0.67, 0.02, 1000),
 'gamma2': np.random.normal(2.37, 0.06, 1000),
 'p1': np.random.normal(3.67, 0.09, 1000),
 'p2': np.random.normal(-2.92, 0.14, 1000),
 'zstar': np.random.normal(2.27, 0.09, 1000),
 'logLa': np.random.normal(10**0.92, 10**0.11, 1000),
 'alpha': np.random.normal(0.18, 0.01, 1000)
}
##################################################
qsolf_LDDE2_hardpars_integration = \
{
 #'K': np.random.normal(10**-5.72, 10**0.07, 1000),
 'K': 10**-5.72,
 'loglstar': np.random.normal(10**44.09, 10**0.05, 1000),
 'gamma1': np.random.normal(0.73, 0.02, 1000),
 'gamma2': np.random.normal(2.22,  0.06, 1000),
 'p1': np.random.normal(4.34, 0.18, 1000),
 'p2': np.random.normal(-0.30, 0.13, 1000),
 'p3': np.random.normal(-7.33, 0.62, 1000),
 'beta1': np.random.normal(-0.19, 0.09, 1000),
 'zstar': np.random.normal(1.85, 0.08, 1000),
 'zstarc2': np.random.normal(3.16, 0.10, 1000),
 'logLa': np.random.normal(10**44.78, 10**0.07, 1000),
 'logLa2': np.random.normal(10**44.46, 10**0.17, 1000),
 'alpha': np.random.normal(0.23, 0.01, 1000),
 'alpha2': np.random.normal(0.12, 0.02, 1000)
}

qsolf_LDDE1_hardpars_integration = \
{
 #'K': np.random.normal(10**-5.63, 10**0.07, 1000),
 'K': 10**-5.63,
 'loglstar': np.random.normal(10**44.10, 10**0.05, 1000),
 'gamma1': np.random.normal(0.72, 0.02, 1000),
 'gamma2': np.random.normal(2.26, 0.07, 1000),
 'p1': np.random.normal(3.97, 0.17, 1000),
 'p2': np.random.normal(-2.08, 0.17, 1000),
 'zstar': np.random.normal(2.02, 0.09, 1000),
 'logLa': np.random.normal(10**44.71, 10**0.09, 1000),
 'alpha': np.random.normal(0.20, 0.01, 1000)
}


#K is not using the random samples
def _LuminosityFunction_LDDE1_integrate(Lx, z, K = None, loglstar = None, gamma1 = None, \
gamma2 = None, p1 = None, p2 = None, zstar = None, logLa = None, alpha = None, **kawargs):
    
        
    if Lx < logLa:
        zc1 = zstar*(Lx / logLa)**alpha
    elif Lx >= logLa:
        zc1 = zstar
    
    #print zc1
##################################################   

    if z <= zc1:
        ex = (1+z)**p1
    elif z > zc1:
        ex = (1+zc1)**p1*((1+z)/(1+zc1))**p2
        
##################################################  
           
    p = K * ((Lx / loglstar)**gamma1 + (Lx / loglstar)**gamma2)**-1 * ex     
    return p
   

#-------------------------------------------------

#K is not using the random samples
def _LuminosityFunction_LDDE2_integrate(Lx, z, K = None, loglstar = None, gamma1 = None, gamma2 = None,\
p1 = None, p2 = None, p3 = None, beta1 = None, zstar = None, zstarc2 = None, logLa = None, \
logLa2 = None, alpha = None, alpha2 = None, **kwargs):
    
    """This equation is taken from Aird et al. (2015) and is based off of Ueda et al. (2014) LDDE equation. 
    This equation stands from the Luminosity Dependent Desnity Equation (LDDE) used to describe the 
    evolutoion of X-ray luminosity from AGNs on the basis of X-ray surveys. The X-ray luminosity Function 
    (XLF) is best described using the LDDE model. The LDDE Model is a function of redshift and X-ray 
    Luminosity best fitting within the range of 0-5 and 10**42-10**48, respectively. This model is the 
    modified version of Ueda et al. (2003) LDDE model. This model accounts for the decay in the comoving 
    numebr density of luminous AGN. """
       
    
            
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
        
##################################################  
    
    p = K * ((Lx / loglstar)**gamma1 + (Lx / loglstar)**gamma2)**-1 * ex
  
    return p
    
#------------------------------------------------