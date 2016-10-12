"""

FluctuatingBackground.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Mon Oct 10 14:29:54 PDT 2016

Description: 

"""

import numpy as np
from types import FunctionType
from ..util import ParameterFile
from ..util.Math import LinearNDInterpolator
from ..populations.Composite import CompositePopulation

class FluctuatingBackground(object):
    def __init__(self, **kwargs):
        """
        Initialize a FluctuatingBackground object.
        
        Creates an object capable of modeling fields that fluctuate spatially.
            
        """
                
        self._kwargs = kwargs.copy()
        self.pf = ParameterFile(**kwargs)
        
    @property
    def pops(self):
        if not hasattr(self, '_pops'):
            self._pops = CompositePopulation(**self._kwargs).pops
    
        return self._pops
    
    def _Vo(self, dr, R):
        if dr >= (2 * R):
            return 0.0
        else:
            return 4. * np.pi * R**3 / 3. - np.pi * dr * (R**2 - dr**2 / 12.)
            
    def spherical_overlap(self, dr, R):
        if not hasattr(self, '_spherical_overlap'):
            self._spherical_overlap = np.vectorize(self._Vo)
        return self._spherical_overlap(dr, R)
            
    def BubbleFillingFactor(self, z, R=None, popid=0):
        
        pop = self.pops[popid]
        
        if pop.pf['pop_bubble_size_dist'] is None:
            R_b = pop.pf['pop_bubble_size']
            V_b = 4. * np.pi * R_b**3 / 3.
            n_b = self.BubbleDensity(z)
            
            return 1. - np.exp(-n_b * V_b)
        
    def BubbleDensity(self, z, R=None, popid=0):
        """
        Compute the volume density of bubbles at redshift z of given radius.
        """
        
        pop = self.pops[popid]
        
        # Can separate size and density artificially
        b_size = pop.pf['pop_bubble_size']
        b_dens = pop.pf['pop_bubble_density']
        
        # This takes care of both dimensions
        b_dist = pop.pf['pop_bubble_size_dist']
        
        # In this case, compute the bubble size distribution from a 
        # user-supplied function, the halo mass function, or excursion set 
        if b_dist is not None:
            assert R is not None        
            
            # Use a user-supplied function for the BSD
            if type(b_dist) is FunctionType:
                return b_dist(z, R)
            # Otherwise, take from hmf or excursion set
            elif type(b_dist) == 'hmf':
                raise NotImplementedError('help')
                # Eventually, distinct from HMF or from excursion set
                # Assume     
                halos = pop.halos
                
            else:
                raise NotImplementedError('help')

        # In this case, there is no bubble size distribution.
        # The density of bubbles is either given as a constant, a user-defined
        # function, or determined from the HMF.
        else:

            if type(b_dens) in [int, float]:
                return b_dens
            elif type(b_dens) is FunctionType:
                return b_dens(z, R)
            elif b_dens == 'hmf':
                halos = pop.halos

                logMmin = np.interp(z, halos.z, np.log10(pop.Mmin))
                n = LinearNDInterpolator([halos.z, halos.logM], halos.ngtm)

                return n([z, logMmin])
        
        raise ValueError('Somethings not right')
        
    def BubbleSizeDistribution(self, z, R=None, popid=0):
        pop = self.pops[popid]
        
        if pop.pf['pop_bubble_size_dist'] is None:
            if pop.pf['pop_bubble_density'] is not None:
                return pop.pf['pop_bubble_density']
            
        raise NotImplementedError('can only handle const. Rbubb right now.')

    def IonizationProbability(self, z, dr, popid=0):
        """
        Compute the probability that two points are both ionized.
    
        Parameters
        ----------
        z : int, float
            Redshift of interest.
        dr : int, float, np.ndarray
            Separation of two points in Mpc.
    
        """
        
        pop = self.pops[popid]
    
        if pop.pf['pop_bubble_size_dist'] is None:
            R = pop.pf['pop_bubble_size']
    
            V = 4. * np.pi * R**3 / 3.
            V_o = self.spherical_overlap(dr, R)
    
            # Abundance of halos
            n_b = self.BubbleDensity(z)
    
            # One and two halo terms, respectively
            oht = (1. - np.exp(-n_b * V_o))
            tht = np.exp(-n_b * V_o) * (1. - np.exp(-n_b * (V - V_o)))**2
    
            return oht, tht
    
        else:
            raise NotImplementedError('help')
    
    def CorrelationFunction(self, z, k, field_1, field_2, popid=0):
        
        # Ionization auto-correlation function
        if field_1 and field_2 == 'h_2':
            
            dr = 2. * np.pi / k
            
            Pii_oh, Pii_th = self.IonizationProbability(z, dr, popid)
            Qi  = self.BubbleFillingFactor(z)
    
            Pii = Pii_oh + Pii_th
                
            return Pii - Qi**2
        
        else:
            raise NotImplementedError('sorry!')
        
    def PowerSpectrum(self, z, k, field_1, field_2, popid=0):
        """
        Return the power spectrum for given input fields at redshift z and
        wavenumber k.
        """
        corr = self.CorrelationFunction(z, k, field_1, field_2, popid)
        
        return np.sqrt(np.fft.fftshift(np.fft.ifft(corr))**2)
            
        