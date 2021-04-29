"""

GaussianSignal.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Jul 23 16:47:26 MDT 2015

Description: 

"""

import numpy as np
from ..util import ParameterFile
from ..physics.Constants import nu_0_mhz
from ..analysis.Global21cm import Global21cm
from ..util.SetDefaultParameterValues import GaussianParameters

# Default parameters
gauss_kwargs = GaussianParameters()

gauss_pars = ['gaussian_A', 'gaussian_nu', 'gaussian_sigma',\
    'gaussian_bias_temp']

def gauss_generic(nu, A, nu0, sigma, offset):
    return (A * np.exp(-(nu - nu0)**2 / 2. / sigma**2) + offset)

class Gaussian21cm(object):
    def __init__(self, **kwargs):
        self.pf = ParameterFile(**kwargs)
    
    def __call__(self, nu, **kwargs):
        """
        Check "gauss_kwargs" for list of acceptable parameters.
    
        Returns
        -------
        glorb.analysis.-21cm instance, which contains the entire signal
        and the turning points conveniently in the "turning_points" 
        attribute.
    
        """
    
        kw = gauss_kwargs.copy()
    
        for element in kwargs:
            kw.update({element: kwargs[element]})
    
        theta = [kw[par] for par in gauss_pars]
    
        return self.gauss_model_for_emcee(nu, theta)
    
    def gauss_model_for_emcee(self, nu, theta):
        """
        Compute 21-cm signal given Gaussian model parameters.
    
        Input parameters assumed to be in the following order:
    
        A, nu, sigma, bias_temp
    
        where Jref, Tref, and xref are the step heights, zref_? are the step
        locations, and dz_? are the step-widths.
    
        Note that Jref is in units of J21.
    
        Returns
        -------
        ares.analysis.Global21cm instance, which contains the entire signal
        and the turning points conveniently in the "turning_points" 
        attribute.
    
        """
    
        # Unpack parameters
        A, nu0, sigma, offset = theta
        dTb = gauss_generic(nu, A, nu0, sigma, offset)
    
        # Save some stuff
        hist = \
        {
         'z': nu_0_mhz / nu - 1.,
         'igm_dTb': dTb,
         'dTb': dTb,
        }
    
        #tmp = Global21cm(data=hist)
        
        return hist
    
