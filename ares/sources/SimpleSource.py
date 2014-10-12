"""

SimpleSource.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Jul  8 13:12:49 MDT 2013

Description: 

"""

import numpy as np
from ..physics.Constants import erg_per_ev

class SimpleSource(object):
    """ Class for creation and manipulation of toy-model radiation sources. """
    def __init__(self, pf, src_pars, spec_pars):
        """ 
        Create toy-model radiation source object.
        
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
        
        self._name = "SimpleSource"
        
        self.Q = self.src_pars['qdot']
        self.E = np.array(self.spec_pars['E'])
        self.LE = np.array(self.spec_pars['LE'])
        self.Lbol = self.Q / (np.sum(self.LE / self.E / erg_per_ev))
        self.Nfreq = len(self.E)

    def SourceOn(self, t):
        return True

    def Luminosity(self, t=None):
        return self.Lbol
        
    def _Intensity(self, E=None, i=None, t=None):
        """
        Return quantity *proportional* to fraction of bolometric luminosity emitted
        at photon energy E.  Normalization handled separately.
        """
        
        return self.LE
        
    def _NormalizeSpectrum(self):
        return np.ones_like(self.E) * self.Lbol
        

