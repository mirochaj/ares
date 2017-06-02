"""

SingleStarModel.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Thu Jun  1 14:52:18 PDT 2017

Description: These are spectra from, e.g., Schaerer (2002), where we get a
series of Q values in different energy bins. Need to construct a continuous
spectrum to be used in RT calculations.

"""

import numpy as np
from .Star import Star
from .Source import Source
from scipy.integrate import quad
from ..util.ReadData import read_lit
from ..physics.Constants import erg_per_ev
from ..util.ParameterFile import ParameterFile

class StarQS(Source):
    def __init__(self, **kwargs):
        self.pf = ParameterFile(**kwargs)

    @property
    def ideal(self):
        if not hasattr(self, '_ideal'):
            self._ideal = Star(source_temperature=self.Teff,
                source_Emin=1, source_Emax=2e2)
        return self._ideal
                    
    @property
    def name(self):
        return self.pf['source_sed']
            
    @property
    def litinst(self):
        if not hasattr(self, '_litinst'):
            self._litinst = read_lit(self.name)
                
        return self._litinst
        
    @property
    def bands(self):
        if not hasattr(self, '_bands'):
            if self.name == 'schaerer2002':
                self._bands = [(11.2, 13.6), (13.6, 24.6), (24.6, 54.4), (54.4, 2e2)]
            else:
                raise NotImplemented('help')
            
        return self._bands
    
    @property
    def bandwidths(self):
        if not hasattr(self, '_bandwidths'):
            self._bandwidths = np.diff(self.bands, axis=1).squeeze()
        return self._bandwidths
    
    @property
    def Elo(self):
        if not hasattr(self, '_Elo'):
            self._Elo = []
            for i, pt in enumerate(self.bands):
                self._Elo.append(pt[0])
            self._Elo = np.array(self._Elo)
    
        return self._Elo
        
    @property
    def Eavg(self):
        if not hasattr(self, '_Eavg'):
            self._Eavg = []
            for i, pt in enumerate(self.bands):
                self._Eavg.append(self.ideal.AveragePhotonEnergy(*pt))
            self._Eavg = np.array(self._Eavg)
            
        return self._Eavg

    @property
    def Q(self):
        if not hasattr(self, '_Q'):
            self._load()
        return self._Q
    
    @Q.setter
    def Q(self, value):
        self._Q = value
    
    @property
    def I(self):
        return self.Q * self.Eavg * erg_per_ev
    
    @property
    def Teff(self):
        if not hasattr(self, '_Teff'):
            self._load()
    
        return self._Teff
    
    @Teff.setter
    def Teff(self, value):
        self._Teff = value    
    
    @property
    def lifetime(self):
        if not hasattr(self, '_lifetime'):
            self._load()
            
        return self._lifetime
        
    @lifetime.setter
    def lifetime(self, value):
        self._lifetime = value    
    
    def _load(self):
        self._Q, self._Teff, self.lifetime = self.litinst._load(**self.pf)
    
    @property
    def norm(self):
        if not hasattr(self, '_norm'):
            
            if self.pf['source_piecewise']:
                self._norm = []
                for i, band in enumerate(self.bands):
                    Q = quad(lambda E: self.ideal.Spectrum(E) / E, *band)[0]
                    self._norm.append(self.Q[i] / Q)
                self._norm = np.array(self._norm)  
            else:
                band = (13.6, self.pf['source_Emax'])
                Q = quad(lambda E: self.ideal.Spectrum(E) / E, *band)[0]
                self._norm = np.array([np.sum(self.Q[1:]) / Q]*4)
            
        return self._norm
        
    def _Spectrum(self, E):
        #for i, band in enumerate(self.bands):
        #    if band[0] <= E < band[1]:
        
        if E < self.bands[0][1]:
            return self.norm[0] * self.ideal.Spectrum(E)
        elif self.bands[1][0] <= E < self.bands[1][1]:
            return self.norm[1] * self.ideal.Spectrum(E)    
        elif self.bands[2][0] <= E < self.bands[2][1]:
            return self.norm[2] * self.ideal.Spectrum(E)
        else:
            return self.norm[3] * self.ideal.Spectrum(E)
            
    @property    
    def _spec_f(self):
        if not hasattr(self, '_spec_f_'):
            self._spec_f_ = np.vectorize(self._Spectrum)
        return self._spec_f_
        
    def Spectrum(self, E, t=None):
        """
        Return quantity *proportional* to fraction of bolometric luminosity.
    
        Parameters
        ----------
        E : int, float
            Photon energy of interest [eV]
    
        """
        
        return self._spec_f(E)
        