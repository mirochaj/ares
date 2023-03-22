"""

RateCoefficients.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Dec 26 20:59:24 2012

Description: Rate coefficients for hydrogen and helium.  Currently using
Fukugita & Kawasaki (1994). Would be nice to include rates from other sources.

"""

import numpy as np
from scipy.misc import derivative
from ..util.Math import interp1d
from ..util.Math import central_difference

T = None
rate_sources = ['fk94']

class RateCoefficients(object):
    def __init__(self, grid=None, rate_src='fk94', T=T, recombination='B',
        interp_rc='linear'):
        """
        Parameters
        ----------
        grid : rt1d.static.Grid instance
        source : str
            fk94 (Fukugita & Kawasaki 1994)
            chianti
        """
        
        self.grid = grid
        self.rate_src = rate_src
        self.interp_rc = interp_rc
        self.T = T

        self.rec = recombination
        
        self.Tarr = 10**np.arange(-1, 6.1, 0.1)
                
        if rate_src not in rate_sources:
            raise ValueError(('Unrecognized rate coefficient source ' +\
                '\'{!s}\'').format(rate_src))
        
    def CollisionalIonizationRate(self, species, T):
        """
        Collisional ionization rate which we denote elsewhere as Beta.
        """    
        
        if self.rate_src == 'fk94':
            if species == 0:  
                return 5.85e-11 * np.sqrt(T) * (1. + np.sqrt(T / 1e5))**-1. \
                    * np.exp(-1.578e5 / T)
              
            if species == 1:    
                return 2.38e-11 * np.sqrt(T) * (1. + np.sqrt(T / 1e5))**-1. \
                    * np.exp(-2.853e5 / T) 
            
            if species == 2:
                return 5.68e-12 * np.sqrt(T) * (1. + np.sqrt(T / 1e5))**-1. \
                    * np.exp(-6.315e5 / T)
        
        else:
            name = self.grid.neutrals[species]
            return self.neutrals[name]['ionizRate'](T)        
           
    @property 
    def _dCollisionalIonizationRate(self):
        if not hasattr(self, '_dCollisionalIonizationRate_'):
            self._dCollisionalIonizationRate_ = {}
            for i, absorber in enumerate(self.grid.absorbers):
                tmp = derivative(lambda T: self.CollisionalIonizationRate(i, T), self.Tarr)
                self._dCollisionalIonizationRate_[i] = interp1d(self.Tarr, tmp, 
                    kind=self.interp_rc)
            
        return self._dCollisionalIonizationRate_
            
    def dCollisionalIonizationRate(self, species, T):
        if self.rate_src == 'fk94':
            return self._dCollisionalIonizationRate[species](T)
            #return derivative(lambda T: self.CollisionalIonizationRate(species, T), T)
        else:
            name = self.grid.neutrals[species]
            return self.neutrals[name]['dionizRate']

    def RadiativeRecombinationRate(self, species, T):
        """
        Coefficient for radiative recombination.  Here, species = 0, 1, 2
        refers to HII, HeII, and HeIII.
        """
        
        if self.rec == 0:
            return np.zeros_like(T)
        
        if self.rate_src == 'fk94':
            if self.rec == 'A':
                if species == 0:
                    return 6.28e-11 * T**-0.5 * (T / 1e3)**-0.2 * (1. + (T / 1e6)**0.7)**-1.
                elif species == 1:
                    return 1.5e-10 * T**-0.6353
                elif species == 2:
                    return 3.36e-10 * T**-0.5 * (T / 1e3)**-0.2 * (1. + (T / 4e6)**0.7)**-1.
            elif self.rec == 'B':
                if species == 0:
                    return 2.6e-13 * (T / 1.e4)**-0.85 
                elif species == 1:
                    return 9.94e-11 * T**-0.6687
                elif species == 2:
                    alpha = 3.36e-10 * T**-0.5 * (T / 1e3)**-0.2 * (1. + (T / 4.e6)**0.7)**-1 # To n >= 1
                                        
                    if type(T) in [float, np.float64]:
                        if T < 2.2e4:
                            alpha *= (1.11 - 0.044 * np.log(T))   # To n >= 2
                        else:
                            alpha *= (1.43 - 0.076 * np.log(T)) # To n >= 2
                    else:
                        alpha[T < 2.2e4] *= (1.11 - 0.044 * np.log(T[T < 2.2e4]))   # To n >= 2
                        alpha[T >= 2.2e4] *= (1.43 - 0.076 * np.log(T[T >= 2.2e4])) # To n >= 2
                        
                    return alpha
            else:
                raise ValueError('Unrecognized RecombinationMethod.  Should be A or B.')
        
        else:
            name = self.grid.ions[species]
            return self.ions[name]['recombRate'](T)
    
    @property 
    def _dRadiativeRecombinationRate(self):
        if not hasattr(self, '_dRadiativeRecombinationRate_'):
            self._dRadiativeRecombinationRate_ = {}
            for i, absorber in enumerate(self.grid.absorbers):
                tmp = derivative(lambda T: self.RadiativeRecombinationRate(i, T), self.Tarr)
                                
                self._dRadiativeRecombinationRate_[i] = interp1d(self.Tarr, tmp, 
                    kind=self.interp_rc)
    
        return self._dRadiativeRecombinationRate_        
    
    def dRadiativeRecombinationRate(self, species, T):
        if self.rate_src == 'fk94':
            return self._dRadiativeRecombinationRate[species](T)
            #return derivative(lambda T: self.RadiativeRecombinationRate(species, T), T)        
        else:
            name = self.ions.neutrals[species] 
            return self.ions[name]['drecombRate']
            
    def DielectricRecombinationRate(self, T):
        """
        Dielectric recombination coefficient for helium.
        """
        
        if self.rate_src == 'fk94':
            return 1.9e-3 * T**-1.5 * np.exp(-4.7e5 / T) * (1. + 0.3 * np.exp(-9.4e4 / T))
        else:
            raise NotImplementedError()
            
    @property 
    def _dDielectricRecombinationRate(self):
        if not hasattr(self, '_dDielectricRecombinationRate_'):
            self._dDielectricRecombinationRate_ = {}
            tmp = derivative(lambda T: self.DielectricRecombinationRate(T), self.Tarr)
            self._dDielectricRecombinationRate_ = interp1d(self.Tarr, tmp, 
                kind=self.interp_rc)
    
        return self._dDielectricRecombinationRate_
                    
    def dDielectricRecombinationRate(self, T):
        if self.rate_src == 'fk94':
            return self._dDielectricRecombinationRate(T)
            #return derivative(self.DielectricRecombinationRate, T)
        else:
            raise NotImplementedError()
            
    def CollisionalIonizationCoolingRate(self, species, T):
        """
        Returns coefficient for cooling by collisional ionization.  These are equations B4.1a, b, and d respectively
        from FK96.
        
            units: erg cm^3 / s
        """

        if self.rate_src == 'fk94':
            if species == 0:
                return 1.27e-21 * np.sqrt(T) * (1. + np.sqrt(T / 1e5))**-1. * np.exp(-1.58e5 / T)
            if species == 1: 
                return 9.38e-22 * np.sqrt(T) * (1. + np.sqrt(T / 1e5))**-1. * np.exp(-2.85e5 / T)
            if species == 2: 
                return 4.95e-22 * np.sqrt(T) * (1. + np.sqrt(T / 1e5))**-1. * np.exp(-6.31e5 / T)
        else:
            raise NotImplemented('Cannot do cooling for rate_source != fk94 (yet).')
    
    @property 
    def _dCollisionalIonizationCoolingRate(self):
        if not hasattr(self, '_dCollisionalIonizationCoolingRate_'):
            self._dCollisionalIonizationCoolingRate_ = {}
            for i, absorber in enumerate(self.grid.absorbers):
                tmp = derivative(lambda T: self.CollisionalExcitationCoolingRate(i, T), self.Tarr)
                self._dCollisionalIonizationCoolingRate_[i] = interp1d(self.Tarr, tmp, 
                    kind=self.interp_rc)
    
        return self._dCollisionalIonizationCoolingRate_
    
    def dCollisionalIonizationCoolingRate(self, species, T):
        if self.rate_src == 'fk94':
            return self._dCollisionalIonizationCoolingRate[species](T)
            #return derivative(lambda T: self.CollisionalIonizationCoolingRate(species, T), T)        
        else:
            raise NotImplementedError()   
           
    def CollisionalExcitationCoolingRate(self, species, T):
        """
        Returns coefficient for cooling by collisional excitation.  These are equations B4.3a, b, and c respectively
        from FK96.
        
            units: erg cm^3 / s
        """
        
        if self.rate_src == 'fk94':
            if species == 0: 
                return 7.5e-19 * (1. + np.sqrt(T / 1e5))**-1. * np.exp(-1.18e5 / T)
            if species == 1: 
                return 9.1e-27 * T**-0.1687 * (1. + np.sqrt(T / 1e5))**-1. * np.exp(-1.31e4 / T)   # CONFUSION
            if species == 2: 
                return 5.54e-17 * T**-0.397 * (1. + np.sqrt(T / 1e5))**-1. * np.exp(-4.73e5 / T)    
        else:
            raise NotImplemented('Cannot do cooling for rate_source != fk94 (yet).')

    @property 
    def _dCollisionalExcitationCoolingRate(self):
        if not hasattr(self, '_dCollisionalExcitationCoolingRate_'):
            self._dCollisionalExcitationCoolingRate_ = {}
            for i, absorber in enumerate(self.grid.absorbers):
                tmp = derivative(lambda T: self.CollisionalExcitationCoolingRate(i, T), self.Tarr)
                self._dCollisionalExcitationCoolingRate_[i] = interp1d(self.Tarr, tmp, 
                    kind=self.interp_rc)
    
        return self._dCollisionalExcitationCoolingRate_
    
    def dCollisionalExcitationCoolingRate(self, species, T):
        if self.rate_src == 'fk94':
            return self._dCollisionalExcitationCoolingRate[species](T)
            #return derivative(lambda T: self.CollisionalExcitationCoolingRate(species, T), T)        
        else:
            raise NotImplementedError()
            
    def RecombinationCoolingRate(self, species, T):
        """
        Returns coefficient for cooling by recombination.  These are equations B4.2a, b, and d respectively
        from FK96.
        
            units: erg cm^3 / s
        """
        
        if self.rec == 0:
            return np.zeros_like(T)
        
        if self.rate_src == 'fk94':
            if species == 0: 
                return 6.5e-27 * np.sqrt(T) * (T / 1e3)**-0.2 * (1.0 + (T / 1e6)**0.7)**-1.0
            if species == 1: 
                return 1.55e-26 * T**0.3647
            if species == 2: 
                return 3.48e-26 * np.sqrt(T) * (T / 1e3)**-0.2 * (1. + (T / 4e6)**0.7)**-1.
        else:
            raise NotImplemented('Cannot do cooling for rate_source != fk94 (yet).')
    
    @property 
    def _dRecombinationCoolingRate(self):
        if not hasattr(self, '_dRecombinationCoolingRate_'):
            self._dRecombinationCoolingRate_ = {}
            for i, absorber in enumerate(self.grid.absorbers):
                tmp = derivative(lambda T: self.RecombinationCoolingRate(i, T), self.Tarr)
                self._dRecombinationCoolingRate_[i] = interp1d(self.Tarr, tmp, 
                    kind=self.interp_rc)
    
        return self._dRecombinationCoolingRate_

    def dRecombinationCoolingRate(self, species, T):
        if self.rate_src == 'fk94':
            return self._dRecombinationCoolingRate[species](T)
            #return derivative(lambda T: self.RecombinationCoolingRate(species, T), T)
        else:
            raise NotImplemented('Cannot do cooling for rate_source != fk94 (yet).')
            
    def DielectricRecombinationCoolingRate(self, T):
        """
        Returns coefficient for cooling by dielectric recombination.  This is equation B4.2c from FK96.
        
            units: erg cm^3 / s
        """
        
        if self.rate_src == 'fk94':
            return 1.24e-13 * T**-1.5 * np.exp(-4.7e5 / T) * (1. + 0.3 * np.exp(-9.4e4 / T))
        else:
            raise NotImplementedError()
            
    @property 
    def _dDielectricRecombinationCoolingRate(self):
        if not hasattr(self, '_dDielectricRecombinationCoolingRate_'):
            tmp = derivative(lambda T: self.DielectricRecombinationCoolingRate(T), self.Tarr)
            self._dDielectricRecombinationCoolingRate_ = interp1d(self.Tarr, tmp, 
                kind=self.interp_rc)
    
        return self._dDielectricRecombinationCoolingRate_        
            
    def dDielectricRecombinationCoolingRate(self, T):
        if self.rate_src == 'fk94':
            return self._dDielectricRecombinationCoolingRate(T)
            #return derivative(self.DielectricRecombinationCoolingRate, T)
        else:
            raise NotImplementedError()
            
            
