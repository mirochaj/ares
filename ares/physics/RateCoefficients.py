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
from scipy.interpolate import interp1d

try:
    import chianti.core as cc
    have_chianti = True        
    T = np.logspace(2, 6, 500)
except ImportError:
    T = None
    have_chianti = False

def central_difference(x, y):
    """
    Compute the derivative of y with respect to x via central difference.

    Parameters
    ----------
    x : np.ndarray
        Array of x values
    y : np.ndarray
        Array of y values

    Returns
    -------
    Tuple containing x values and corresponding y derivatives.

    """    

    dydx = ((np.roll(y, -1) - np.roll(y, 1)) \
        / (np.roll(x, -1) - np.roll(x, 1)))[1:-1]

    return x[1:-1], dydx
    
rate_sources = ['fk94', 'chianti']

class RateCoefficients:
    def __init__(self, grid=None, rate_src='fk94', T=T, recombination='B'):
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
        self.T = T

        self.rec = recombination
            
        if self.rate_src == 'chianti':
            if not have_chianti:
                raise ValueError('ChiantiPy not found.')
            else:    
                self._init_chianti()
                
        if rate_src not in rate_sources:
            raise ValueError('Unrecognized rate coefficient source \'%s\'' % rate_src)
        
    def _init_chianti(self):  
        """
        Create lookup tables for Chianti atomic database rates.
        """      
        
        self.ions = {}
        self.neutrals = {}
        
        for neutral in self.grid.neutrals:
            atom = cc.ion(neutral, temperature=self.T)
            atom.ionizRate()
            
            self.neutrals[neutral] = {}
            
            self.neutrals[neutral]['ionizRate'] = \
                interp1d(self.T, atom.IonizRate['rate'], kind='cubic')
                
            T, dRdT = central_difference(self.T, atom.IonizRate['rate'])
            self.neutrals[neutral]['dionizRate'] = \
                interp1d(T, dRdT, kind='cubic')    
                
        for ion in self.grid.ions:
            atom = cc.ion(ion, temperature=self.T)
            atom.recombRate()
            
            self.ions[ion] = {}
            
            self.ions[ion]['recombRate'] = \
                interp1d(self.T, atom.RecombRate['rate'], kind='cubic')  
            
            T, dRdT = central_difference(self.T, atom.RecombRate['rate'])
            self.ions[ion]['drecombRate'] = \
                interp1d(T, dRdT, kind='cubic')          
        
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
            
    def dCollisionalIonizationRate(self, species, T):
        if self.rate_src == 'fk94':
            return derivative(lambda T: self.CollisionalIonizationRate(species, T), T)
        else:
            name = self.grid.neutrals[species]
            return self.neutrals[name]['dionizRate']

    def RadiativeRecombinationRate(self, species, T):
        """
        Coefficient for radiative recombination.  Here, species = 0, 1, 2
        refers to HII, HeII, and HeIII.
        """
        
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
                    
                    if type(T) in [np.float64]:
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
            
    def dRadiativeRecombinationRate(self, species, T):
        if self.rate_src == 'fk94':
            return derivative(lambda T: self.RadiativeRecombinationRate(species, T), T)        
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
            
    def dDielectricRecombinationRate(self, T):
        if self.rate_src == 'fk94':
            return derivative(self.DielectricRecombinationRate, T)
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
    
    def dCollisionalIonizationCoolingRate(self, species, T):
        if self.rate_src == 'fk94':
            return derivative(lambda T: self.CollisionalIonizationCoolingRate(species, T), T)        
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

    def dCollisionalExcitationCoolingRate(self, species, T):
        if self.rate_src == 'fk94':
            return derivative(lambda T: self.CollisionalExcitationCoolingRate(species, T), T)        
        else:
            raise NotImplementedError()
            
    def RecombinationCoolingRate(self, species, T):
        """
        Returns coefficient for cooling by recombination.  These are equations B4.2a, b, and d respectively
        from FK96.
        
            units: erg cm^3 / s
        """
        
        if self.rate_src == 'fk94':
            if species == 0: 
                return 6.5e-27 * np.sqrt(T) * (T / 1e3)**-0.2 * (1.0 + (T / 1e6)**0.7)**-1.0
            if species == 1: 
                return 1.55e-26 * T**0.3647
            if species == 2: 
                return 3.48e-26 * np.sqrt(T) * (T / 1e3)**-0.2 * (1. + (T / 4e6)**0.7)**-1.
        else:
            raise NotImplemented('Cannot do cooling for rate_source != fk94 (yet).')

    def dRecombinationCoolingRate(self, species, T):
        if self.rate_src == 'fk94':
            return derivative(lambda T: self.RecombinationCoolingRate(species, T), T)
        else:
            raise NotImplementedError()
            
    def DielectricRecombinationCoolingRate(self, T):
        """
        Returns coefficient for cooling by dielectric recombination.  This is equation B4.2c from FK96.
        
            units: erg cm^3 / s
        """
        
        if self.rate_src == 'fk94':
            return 1.24e-13 * T**-1.5 * np.exp(-4.7e5 / T) * (1. + 0.3 * np.exp(-9.4e4 / T))
        else:
            raise NotImplementedError()
            
    def dDielectricRecombinationCoolingRate(self, T):
        if self.rate_src == 'fk94':
            return derivative(self.DielectricRecombinationCoolingRate, T)
        else:
            raise NotImplementedError()