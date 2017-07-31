"""

FluctuatingBackground.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Mon Oct 10 14:29:54 PDT 2016

Description: 

"""

import numpy as np
from types import FunctionType
from ..physics import Cosmology
from ..util import ParameterFile
from scipy.special import erfinv
from ..physics.Constants import g_per_msun
from ..util.Math import LinearNDInterpolator
from ..populations.Composite import CompositePopulation

class FluctuatingBackground(object):
    def __init__(self, grid=None, **kwargs):
        """
        Initialize a FluctuatingBackground object.
        
        Creates an object capable of modeling fields that fluctuate spatially.
            
        """
                
        self._kwargs = kwargs.copy()
        self.pf = ParameterFile(**kwargs)
        
        # Some useful physics modules
        if grid is not None:
            self.grid = grid
            self.cosm = grid.cosm
        else:
            self.grid = None
            self.cosm = Cosmology()
        
    @property
    def pops(self):
        if not hasattr(self, '_pops'):
            self._pops = CompositePopulation(**self._kwargs).pops
    
        return self._pops
    
    def _Vo_sphere(self, dr, R):
        if dr >= (2 * R):
            return 0.0
        else:
            return 4. * np.pi * R**3 / 3. - np.pi * dr * (R**2 - dr**2 / 12.)
        
    def _Vo_shell(self, dr, R, D):
        if dr >= (2 * (R + D)):
            return 0.0
        else:

            Rs = R + D
            
            # Full overlap region of two spheres the size of our 
            # bubble plus its shell
            reg1 = 4. * np.pi * Rs**3 / 3. - np.pi * dr * (Rs**2 - dr**2 / 12.)
            
            # The volume of the region we need to subtract off to exclude
            # parts of the overlap region where one or both of the points
            # would be ionized
            reg2 = np.pi * (Rs + R - dr)**2 \
                * (dr**2 + 2. * dr * R - 3. * R**2 + 2. * dr * Rs + 6. * R * Rs - 3. * Rs**2)
            reg2 /= (12. * dr)
            
            return reg1 - 2. * reg2
            
    def overlap_region_sphere(self, dr, R):
        if not hasattr(self, '_overlap_region_sphere'):
            self._overlap_region_sphere = np.vectorize(self._Vo_sphere)
        return self._overlap_region_sphere(dr, R)
    
    def overlap_region_shell(self, dr, R, D):
        if not hasattr(self, '_overlap_region_shell'):
            self._overlap_region_shell = np.vectorize(self._Vo_shell)
        return self._overlap_region_shell(dr, R, D)    
            
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
            
            QHII = 1. - np.exp(-np.trapz(dndm * V * M, x=np.log(M)))
            
            return QHII
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
        sigma_min = np.interp(pop.Mmin(z) * zeta, pop.halos.M, s)
        return self._delta_c(z) - np.sqrt(2) * self._K(zeta) * sigma_min
    
    def _B1(self, z, zeta=40, popid=0):
        pop = self.pops[popid]
        s = pop.halos.sigma_0
        sigma_min = np.interp(pop.Mmin(z) * zeta, pop.halos.M, s)
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
        
        if not hasattr(self, '_bsd_cache'):
            self._bsd_cache = {}
            
        if z in self._bsd_cache:
            return self._bsd_cache[z]
        
        pop = self.pops[popid]
        
        if pop.pf['pop_bubble_size_dist'] is None:
            if pop.pf['pop_bubble_density'] is not None:
                Rb = pop.pf['pop_bubble_size']
                Mb = (4. * np.pi * Rb**3 / 3.) * pop.cosm.mean_density0 / g_per_msun
                
                self._bsd_cache[z] = Rb, Mb, pop.pf['pop_bubble_density']
                
            else:
                raise NotImplementedError('help')
        elif pop.pf['pop_bubble_size_dist'].lower() == 'fzh04':
            zeta = 40.
            Mb = pop.halos.M * zeta
            rho0 = pop.cosm.mean_density0
            sig = pop.halos.sigma_0
            S = sig**2
            
            pcross = self._B0(z, zeta) / np.sqrt(2. * np.pi * S**3) \
                * np.exp(-0.5 * self._B(z, zeta)**2 / S)
                
            R = ((Mb / rho0) * 0.75 / np.pi)**(1./3.)
            dndm = rho0 * pcross * 2 * np.abs(pop.halos.dlns_dlnm) * S / Mb**2

            self._bsd_cache[z] = R, Mb, dndm
        else:
            raise NotImplementedError('Unrecognized option: %s' % pop.pf['pop_bubble_size_dist'])

        return self._bsd_cache[z]

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
            V_o = self.overlap_region_sphere(dr, R)
    
            # Abundance of halos
            n_b = self.BubbleDensity(z)

            # One and two halo terms, respectively
            if pop.pf['pop_one_halo_term']:
                oht = (1. - np.exp(-n_b * V_o))
            else:
                oht = 0.0
            
            if pop.pf['pop_two_halo_term']:
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

            iz = np.argmin(np.abs(z - pop.halos.z))
            Mmin = pop._tab_Mmin[iz]
            iM = np.argmin(np.abs(pop.halos.M - Mmin))
            
            xi = np.zeros_like(dr)
            for i, sep in enumerate(dr):
                Vo = self.overlap_region_sphere(sep, R)
                
                integrand1 = dndm[iM:] * Vo[iM:]
                exp_int1 = np.exp(-np.trapz(integrand1 * Mb[iM:], 
                    x=np.log(Mb[iM:])))                    
                
                if pop.pf['pop_one_halo_term']:
                    xi[i] += (1. - exp_int1) 
                if pop.pf['pop_two_halo_term']:
                    integrand2 = dndm[iM:] * (V[iM:] - Vo[iM:])
                    if pop.pf['pop_biased']:
                        ep = self.excess_probability(z, dr, popid)
                        integrand2 *= (1. + ep)

                    exp_int2 = np.exp(-np.trapz(integrand2 * Mb[iM:], 
                        x=np.log(Mb[iM:])))
                        
                    xi[i] += exp_int1 * (1. - exp_int2)**2

            return xi

        elif type(pop.pf['pop_bubble_size_dist']) is FunctionType:
            raise NotImplementedError('help')
        else:
            raise NotImplementedError('help')
            
    def HeatedProbability(self, z, dr=None, popid=0):
        """
        Compute the probability that two points are both heated.
    
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
            Rs = pop.pf['pop_bubble_shell_size']
    
            V = 4. * np.pi * (R+Rs)**3 / 3.
            V_o = self.overlap_region_shell(dr, R, Rs)
    
            # Abundance of halos
            n_b = self.BubbleDensity(z)
    
            # One and two halo terms, respectively
            if pop.pf['pop_one_halo_term']:
                oht = (1. - np.exp(-n_b * V_o))
            else:
                oht = 0.0
    
            if pop.pf['pop_two_halo_term']:
                tht = np.exp(-n_b * V_o) * (1. - np.exp(-n_b * (V - V_o)))**2
            else:
                tht = 0.0
    
            return oht + tht
    
        elif pop.pf['pop_bubble_size_dist'].lower() == 'fzh04':
        
            # Should cache this for each redshift.
            R, Mb, dndm = self.BubbleSizeDistribution(z, popid)
            
            Rs = pop.pf['pop_bubble_shell_size']
            Tk = pop.pf['pop_bubble_shell_temp']
    
            # One of these terms will be different if bias of sources
            # is included.
            Vsh = 4. * np.pi * (Rs - R)**3 / 3.
            Vsph = 4. * np.pi * Rs**3 / 3.
    
            iz = np.argmin(np.abs(z - pop.halos.z))
            Mmin = pop._tab_Mmin[iz]
            iM = np.argmin(np.abs(pop.halos.M - Mmin))
    
            TT = np.zeros_like(dr)
            for i, sep in enumerate(dr):
                Vo_sh = self.overlap_region_shell(sep, R, Rs)
                
                integrand1 = dndm[iM:] * Vo_sh[iM:]
                exp_int1 = np.exp(-np.trapz(integrand1 * Mb[iM:], 
                    x=np.log(Mb[iM:])))                    
    
                if pop.pf['pop_one_halo_term']:
                    TT[i] += (1. - exp_int1)
                if pop.pf['pop_two_halo_term']:
                    # In this case, for the two halo term we subtract off
                    # the entire volume of overlapping spheres of radius R+Rs,
                    # since sources in that region will either heat both points
                    # or heat one and ionize the other. We want only one
                    # of the points to be heated here, and the other 
                    # unheated and unionized (a.k.a neutral, smart guy).
                    Vo_sph = self.overlap_region_sphere(sep, R+Rs)
                    
                    integrand2 = dndm[iM:] * (Vsh[iM:] - Vo_sph[iM:])
                    if pop.pf['pop_biased']:
                        ep = self.excess_probability(z, dr, popid)
                        integrand2 *= (1. + ep)
                
                    exp_int2 = np.exp(-np.trapz(integrand2 * Mb[iM:], 
                        x=np.log(Mb[iM:])))
                
                    TT[i] += exp_int1 * (1. - exp_int2)**2
    
            #cc = TT * (self.cosm.TCMB(z) / Tk)**2
    
            return TT
    
        elif type(pop.pf['pop_bubble_size_dist']) is FunctionType:
            raise NotImplementedError('help')
        else:
            raise NotImplementedError('help')
    
    
    #def JointProbability(self, z, dr, zeta, Tprof=None, term='xx'):
    #    """
    #    Compute the joint probability that two points are ionized, heated, etc.
    #    
    #    Parameters
    #    ----------
    #    z : int, float
    #    dr : np.ndarray
    #        Array of scales to consider.
    #    zeta : ionization parameter
    #    Tprof : 
    #    term : str
    #        
    #    """
    #    
    #    pop = self.pops[popid]
    #
    #    if pop.pf['pop_bubble_size_dist'] is None:
    #        R = pop.pf['pop_bubble_size']
    #        Rs = pop.pf['pop_bubble_shell_size']
    #
    #        V = 4. * np.pi * (R+Rs)**3 / 3.
    #        V_o = self.overlap_region_shell(dr, R, Rs)
    #
    #        # Abundance of halos
    #        n_b = self.BubbleDensity(z)
    #
    #        # One and two halo terms, respectively
    #        if pop.pf['pop_one_halo_term']:
    #            oht = (1. - np.exp(-n_b * V_o))
    #        else:
    #            oht = 0.0
    #
    #        if pop.pf['pop_two_halo_term']:
    #            tht = np.exp(-n_b * V_o) * (1. - np.exp(-n_b * (V - V_o)))**2
    #        else:
    #            tht = 0.0
    #
    #        return oht + tht
    #
    def excess_probability(self, z, r, popid=0):
        pop = self.pops[popid]
        
        iz = np.argmin(np.abs(z - pop.halos.z))
        Mmin = pop._tab_Mmin[iz]
        iM = np.argmin(np.abs(pop.halos.M - Mmin))
        
        b = pop.halos.bias(z, pop.halos.logM[iM:]).squeeze()
        bbar = 1.
        
        xi_dd = pop.halos.CorrelationFunction(z, r)
        
        return b * bbar * np.array(xi_dd)

    def CorrelationFunction(self, z, field_1, field_2, dr=None, popid=0):

        # Ionization auto-correlation function
        if field_1 == field_2 == 'x':
            Qi  = self.BubbleFillingFactor(z)
            Pii = self.IonizationProbability(z, dr, popid)
            return Pii - Qi**2
        elif field_1 == field_2 == 'd':
            return pop.halos.CorrelationFunction(z, dr)
        elif field_1 == field_2 == 'c':
            Phh = self.HeatedProbability(z, dr, popid)
            #Phc = self.HeatedProbability(z, dr, popid)
            #Pcc = self.HeatedProbability(z, dr, popid)
            #
            return Phh
        elif field_1 in ['x', 'd'] and field_2 in ['x', 'd']:
            raise NotImplementedError('no cross terms yet')
        elif field_1 in ['x', 'T'] and field_2 in ['x', 'T']:
            raise NotImplementedError('no cross terms yet')    
        elif field_1 in ['d', 'T'] and field_2 in ['d', 'T']:
            raise NotImplementedError('no cross terms yet')    
        else:
            raise NotImplementedError('sorry!')
        
    def PowerSpectrum(self, z, field_1, field_2, k=None, popid=0):
        """
        Return the power spectrum for given input fields at redshift z and
        wavenumber k.
        """
        corr = self.CorrelationFunction(z, field_1, field_2, k=k, popid=popid)

        return np.sqrt(np.fft.fftshift(np.fft.ifft(corr))**2)
            
        