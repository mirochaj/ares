"""

SimpleSource.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Jul  8 13:12:49 MDT 2013

Description: 

"""

import numpy as np
from .Source import Source
from ..physics.Constants import erg_per_ev
from ..util.SetDefaultParameterValues import SourceParameters

class DeltaFunction(Source):
    def __init__(self, **kwargs):
        """ 
        Create delta function radiation source object.
        
        Parameters
        ----------
        pf: dict
            Full parameter file.
        src_pars: dict
            Contains source-specific parameters.
        spec_pars: dict
            Contains spectrum-specific parameters.
            
        """  
        
        Source.__init__(self, **kwargs)
        
        assert self.pf['source_sed'] == 'delta', \
            "Error: source is {}, should be delta!".format(self.pf['source_sed'])
        
        self.E = self.pf['source_Emax']

    def SourceOn(self, t):
        return True

    def _Intensity(self, E=None, i=None, t=None):
        """
        Return quantity *proportional* to fraction of bolometric luminosity emitted
        at photon energy E.  Normalization handled separately.
        """
        
        if E != self.E:
            return 0.0
        else:
            return 1.0    

class Toy(Source):
    """ Class for creation and manipulation of toy-model radiation sources. """
    def __init__(self, **kwargs):
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
        
        Source.__init__(self, **kwargs)

        self.Q = self.pf['source_qdot']
        self.E = np.atleast_1d(self.pf['source_E'])
        self.LE = np.atleast_1d(self.pf['source_LE'])
        self.Lbol = lambda t: self.Q / (np.sum(self.LE / self.E / erg_per_ev))
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
        

