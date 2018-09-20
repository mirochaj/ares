"""

GalaxyEnsemble.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Wed Jul  6 11:02:24 PDT 2016

Description: 

"""

import numpy as np
from ..util.Stats import bin_e2c, bin_c2e
#from .GalaxyCohort import GalaxyCohort
from .Halo import HaloPopulation
#from scipy.interpolate import RectBivariateSpline

class GalaxyEnsemble(HaloPopulation):
    
    def __init__(self, **kwargs):
        # May not actually need this...
        HaloPopulation.__init__(self, **kwargs)
        
    @property
    def dust(self):
        if not hasattr(self, '_dust'):
            self._dust = DustCorrection(**self.pf)
        return self._dust    
    
    @property
    def tab_z(self):
        if not hasattr(self, '_tab_z'):
            h = self.histories
        return self._tab_z
            
    @tab_z.setter
    def tab_z(self, value):
        self._tab_z = value

    @property
    def histories(self):
        if not hasattr(self, '_histories'):
            
            raw = self.pf['pop_histories']
            
            if raw is None:
                raise ValueError('No `pop_histories` provided!')
            elif type(raw) is dict:
                self.tab_z = raw['z']
                self._histories = raw
            else:
                raise NotImplemented('help!')    
                            
        return self._histories
        
    #@history.setter
    #def history(self, value):
    #    pass
    
    def _get_mstell(self):
        """
        If stellar masses haven't been computed, integrate SFRs.
        """
        pass
        
    def LuminosityFunction(self, z, x, mags=True, wave=1600., band=None):
        """
        Compute the luminosity function from discrete histories.
        """
    
        iz = np.argmin(np.abs(z - self.tab_z))
        
        if self.pf['pop_aging']:
            raise NotImplemented('help!')
        
        # All star formation rates at this redshift, 'marginalizing' over
        # formation redshift.
        sfr = self.histories['SFR'][:,iz]
                
        # At the moment, should be redshift independent. Objects springing
        # into existence should just have SFR=0 until they form?
        w = self.histories['w'][:,iz]
                
        # Might need to specify units of the weights
        
        
        # Note: if we knew ahead of time that this was a Cohort population,
        # we could replace ':' with 'iz:' in this slice, since we need
        # not consider objects with formation redshifts < current redshift.
        
        L = self.src.L_per_sfr(wave) * sfr
        
        MAB = self.magsys.L_to_MAB(L, z=z)
        
        #w *= 1. / np.diff(MAB)
        
        # Need to modify weights, since the mapping from Mh -> L -> mag
        # is not linear.
        #w *= np.diff(self.histories['Mh'][:,iz]) / np.diff(L)
        
            
        # Should assert that max(MAB) is < max(MAB)
        
        # If L=0, MAB->inf. Hack these elements off if they exist.
        # This should be a clean cut, i.e., there shouldn't be random
        # spikes where L==0, all L==0 elements should be a contiguous chunk.
        Misinf = np.isinf(MAB)
        Misok  = np.logical_not(Misinf)
        
        #assert min(x) <= min(MAB[Misok])
        #assert max(x) >= max(MAB[Misok])
        
        if np.any(Misinf):
            k = np.max(np.argwhere(Misinf)) + 1
        else:
            k = 0
        
        hist, bin_edges = np.histogram(MAB[k:], weights=w[k:], 
            bins=bin_c2e(x), density=True)
        
        # Need to be more careful with this, since some halos have 0 luminosity.
        # If no aging, could just filter out SFR=0 halos, but in general,
        # that won't be correct.
        if self.pf['pop_aging']:
            raise NotImplemented('help')
        else:    
            on = sfr > 0
        
        #N = np.trapz(self.histories['nh'][on==1,iz], x=Mh[on==1])
        N = np.sum(w[on==1])
            
        return hist * N
        
    def SFRF(self, z):
        pass
    
    def StellarMassFunction(self, z):
        pass
        
    def PDF(self, z, **kwargs):
        # Look at distribution in some quantity at fixed z, potentially other stuff.
        pass
        
        