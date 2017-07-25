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
from scipy.special import erfinv
from ..physics.Constants import g_per_msun
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
        elif pop.pf['pop_bubble_size_dist'].lower() == 'fzh04':
            R, M, dndm = self.BubbleSizeDistribution(z, popid)
            V = 4. * np.pi * R**3 / 3.
            return np.trapz(dndm * V * M, x=np.log(M))
        else:
            raise NotImplemented('Uncrecognized option for BSD.')
        
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
        
    def _K(self, zeta):
        return erfinv(1. - 1. / zeta)
    
    def _delta_c(self, z, popid=0):
        pop = self.pops[popid]
        return pop.cosm.delta_c0 / pop.growth_factor(z)
        
    def _B0(self, z, zeta=40, popid=0):
        pop = self.pops[popid]
        s = pop.halos.sigma_0
        sigma_min = np.interp(pop.Mmin[0] * zeta, pop.halos.M, s)
        return self._delta_c(z) - np.sqrt(2) * self._K(zeta) * sigma_min
    
    def _B1(self, z, zeta=40, popid=0):
        pop = self.pops[popid]
        s = pop.halos.sigma_0
        sigma_min = np.interp(pop.Mmin[0] * zeta, pop.halos.M, s)
        ddx_ds2 = self._K(zeta) / np.sqrt(2. * (sigma_min**2 - s**2))
    
        return ddx_ds2[s == s.min()]
    
    def _B(self, z, zeta=40., popid=0):
        """
        Linear barrier.
        """
        pop = self.pops[popid]
        s = pop.halos.sigma_0
        return self._B0(z, zeta, popid) + self._B1(z, zeta, popid) * s**2
                
    def BubbleSizeDistribution(self, z, popid=0):
        pop = self.pops[popid]
        
        if pop.pf['pop_bubble_size_dist'] is None:
            if pop.pf['pop_bubble_density'] is not None:
                Rb = pop.pf['pop_bubble_size']
                Mb = (4. * np.pi * Rb**3 / 3.) * pop.cosm.mean_density0 / g_per_msun
                return Rb, Mb, pop.pf['pop_bubble_density']
        elif pop.pf['pop_bubble_size_dist'].lower() == 'fzh04':
            zeta = 80.
            Mb = pop.halos.M * zeta
            rho0 = pop.cosm.mean_density0
            sig = pop.halos.sigma_0
            S = sig**2
            
            pcross = self._B0(z, zeta) / np.sqrt(2. * np.pi * S**3) \
                * np.exp(-0.5 * self._B(z, zeta)**2 / S)
                
            R = ((Mb / rho0) * 0.75 / np.pi)**(1./3.)
            dndm = rho0 * pcross * 2 * np.abs(pop.halos.dlns_dlnm) * S / Mb**2

            return R, Mb, dndm
        else:
            raise NotImplementedError('Unrecognized option: %s' % pop.pf['pop_bubble_size_dist'])

    def IonizationProbability(self, z, dr=None, popid=0):
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
            if self.pf['pop_one_halo_term']:
                oht = (1. - np.exp(-n_b * V_o))
            else:
                oht = 0.0
            
            if self.pf['pop_two_halo_term']:
                tht = np.exp(-n_b * V_o) * (1. - np.exp(-n_b * (V - V_o)))**2
            else:
                tht = 0.0

            return oht + tht

        elif pop.pf['pop_bubble_size_dist'].lower() == 'fzh04':

            # Should cache this for each redshift.
            R, Mb, dndm = self.BubbleSizeDistribution(z, popid)
                        
            # One of these terms will be different if bias of sources
            # is included.
            V = 4. * np.pi * R**3 / 3.

            Mmin = 1e8
            iM = np.argmin(np.abs(pop.halos.M - 1e8))
            
            xi = np.zeros_like(dr)
            for i, sep in enumerate(dr):
                Vo = self.spherical_overlap(sep, R)
                
                integrand1 = dndm[iM:] * Vo[iM:]
                exp_int1 = np.exp(-np.trapz(integrand1 * Mb[iM:], 
                    x=np.log(Mb[iM:])))                    
                
                if self.pf['pop_one_halo_term']:
                    xi[i] += (1. - exp_int1) 
                if self.pf['pop_two_halo_term']:
                    integrand2 = dndm[iM:] * (V[iM:] - Vo[iM:])
                    if pop.pf['pop_biased']:
                        bias = pop.halos.bias(z, pop.halos.logM[iM:]).squeeze()
                        integrand2 *= (1. + bias)

                    exp_int2 = np.exp(-np.trapz(integrand2 * Mb[iM:], 
                        x=np.log(Mb[iM:])))
                        
                    xi[i] += exp_int1 * (1. - exp_int2)**2

            return xi

        elif type(pop.pf['pop_bubble_size_dist']) is FunctionType:
            raise NotImplementedError('help')
        else:
            raise NotImplementedError('help')

    def CorrelationFunction(self, z, field_1, field_2, dr=None, popid=0):

        # Ionization auto-correlation function
        if field_1 == field_2 == 'h_2':
            Qi  = self.BubbleFillingFactor(z)
            Pii = self.IonizationProbability(z, dr, popid)

            # Interpolate to linear R grid here?
            return Pii - Qi**2
        elif field_1 == field_2 == 'd':
            pass
        else:
            raise NotImplementedError('sorry!')
        
    def PowerSpectrum(self, z, field_1, field_2, k=None, popid=0):
        """
        Return the power spectrum for given input fields at redshift z and
        wavenumber k.
        """
        corr = self.CorrelationFunction(z, field_1, field_2, k=k, popid=popid)

        return np.sqrt(np.fft.fftshift(np.fft.ifft(corr))**2)
            
        