"""

StellarSource.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Jul  8 09:56:35 MDT 2013

Description: 

"""

import numpy as np
from scipy.integrate import quad
from ..physics.Constants import *

def _Planck(E, T):
    """ Returns specific intensity of blackbody at T."""
    
    nu = E * erg_per_ev / h
    return 2.0 * h * nu**3 / c**2 / (np.exp(h * nu / k_B / T) - 1.0)

class StellarSource(object):
    """ Class for creation and manipulation of stellar radiation sources. """
    def __init__(self, pf, src_pars, spec_pars):
        """ 
        Parameters
        ----------
        pf: dict
            Full parameter file.
        src_pars: dict
            Contains source-specific parameters.
        spec_pars: dict
            Contains spectrum-specific parameters.
            
        """  
        
        self.pf = pf
        self.src_pars = src_pars
        self.spec_pars = spec_pars
        
        self._name = "StellarSource"

        self.Q = self.src_pars['qdot']
        self.T = self.src_pars['temperature']

        # Number of ionizing photons per cm^2 of surface area for BB of 
        # temperature self.T. Use to solve for stellar radius (which we need 
        # to get Lbol).  The factor of pi gets rid of the / sr units
        QNorm = np.pi * 2. * (k_B * self.T)**3 * \
                quad(lambda x: x**2 / (np.exp(x) - 1.), 
                13.6 * erg_per_ev / k_B / self.T, np.inf)[0] \
                / h**3 / c**2             
        # "Analytic" solution has poly-logarithm function - not in scipy (yet)
        self.R = np.sqrt(self.Q / 4. / np.pi / QNorm)        
        self.Lbol = self.Lbol0 = 4. * np.pi * self.R**2 * sigma_SB * self.T**4
        
        self.N = len(self.spec_pars['type'])
        
        self.tau = self.src_pars['lifetime'] * self.pf['time_units']
        
        if self.N != 1:
            raise ValueError('No support for multi-component stellar SEDs.')
        
    def SourceOn(self, t):
        if t < self.tau:
            return True    
        else:
            return False
            
    def Luminosity(self, t=None):
        return self.Lbol
        #self.Q / (np.sum(self.LE / self.E / erg_per_ev))
        
    def _Intensity(self, E, i=None, t=None):
        """
        Return quantity *proportional* to fraction of bolometric luminosity emitted
        at photon energy E.  Normalization handled separately.
        """
        
        return _Planck(E, self.T)
        
    def _NormalizeSpectrum(self):
        integral, err = quad(self._Intensity,
            self.spec_pars['EminNorm'][0], self.spec_pars['EmaxNorm'][0])
        
        return np.array([self.Lbol0 / integral])
