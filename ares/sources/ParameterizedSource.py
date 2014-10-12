"""

ParameterizedSource.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Oct  2 16:54:04 MDT 2013

Description: 

"""

import numpy as np
from scipy.integrate import quad

class ParameterizedSource(object):
    """ Class for creation and manipulation of parameterized radiation sources. """
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
        
        self._name = "ParameterizedSource"

        if self.pf['source_normalized']:
            self.Lbol = self.Lbol0 = 1.0
        else:
            self.Lbol = self.Lbol0 = self.pf['source_Lbol']
        
        self.N = len(self.spec_pars['type'])
        
        if self.N != 1:
            raise ValueError('No support for multi-component stellar SEDs.')
        
    def SourceOn(self, t):
        return True    
        
    def Luminosity(self, t=None):
        return self.Lbol
        #self.Q / (np.sum(self.LE / self.E / erg_per_ev))
        
    def _Intensity(self, E, i=None, t=None):
        """
        Return quantity *proportional* to fraction of bolometric luminosity emitted
        at photon energy E.  Normalization handled separately.
        """
        
        return self.pf['spectrum_function'](E)
        
    def _NormalizeSpectrum(self):
        
        if self.pf['source_normalized']:
            return np.ones(1)
            
        integral, err = quad(self._Intensity,
            self.spec_pars['EminNorm'][0], self.spec_pars['EmaxNorm'][0])
        
        return np.array([self.Lbol0 / integral])
        
        
