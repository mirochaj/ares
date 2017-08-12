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
        
        self.pf = SourceParameters()
        self.pf.update(kwargs)
        
        Source.__init__(self)

        self.Q = self.pf['source_qdot']
        self.E = np.array(self.pf['source_E'])
        self.LE = np.array(self.pf['source_LE'])
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
        

