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
            return 0.0, 0.0
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
            
            # Avoid double-counting for closely separated points
            if dr <= (2 * R):
                reg3 = 4. * np.pi * R**3 / 3. - np.pi * dr * (R**2 - dr**2 / 12.)
                reg2 = reg2 - 0.5 * reg3
            
            return reg1, reg2
            
    def overlap_region_sphere(self, dr, R):
        if not hasattr(self, '_overlap_region_sphere'):
            self._overlap_region_sphere = np.vectorize(self._Vo_sphere)
        return self._overlap_region_sphere(dr, R)
    
    def overlap_region_shell(self, dr, R, D):
        if not hasattr(self, '_overlap_region_shell'):
            self._overlap_region_shell = np.vectorize(self._Vo_shell)
        return self._overlap_region_shell(dr, R, D)    
            
    def BubbleShellFillingFactor(self, z, zeta):
        if self.pf['bubble_size_dist'] is None:
            R_b = self.pf['bubble_size']
            V_b = 4. * np.pi * R_b**3 / 3.
            n_b = self.BubbleDensity(z)
        
            return 1. - np.exp(-n_b * V_b)
        elif self.pf['bubble_size_dist'].lower() == 'fzh04':
            Rb, Mb, dndm = self.BubbleSizeDistribution(z, zeta)
            
            if self.pf['bubble_shell_size_rel'] is not None:
                Rs = Rb * (1. + self.pf['bubble_shell_size_rel'])
            else:
                Rs = Rb + self.pf['bubble_shell_size_abs']
                
            Vsh = 4. * np.pi * (Rs - Rb)**3 / 3.
        
            Qhot = np.trapz(dndm * Vsh * Mb, x=np.log(Mb))
        
            return Qhot
        else:
            raise NotImplemented('Uncrecognized option for BSD.')

    def BubbleFillingFactor(self, z, zeta):
                
        if self.pf['bubble_size_dist'] is None:
            R_b = self.pf['bubble_size']
            V_b = 4. * np.pi * R_b**3 / 3.
            n_b = self.BubbleDensity(z)
            
            return 1. - np.exp(-n_b * V_b)
        elif self.pf['bubble_size_dist'].lower() == 'fzh04':
            
            #return self.pops[0].halos.fcoll
            
            R, M, dndm = self.BubbleSizeDistribution(z, zeta)
            V = 4. * np.pi * R**3 / 3.
            
            dndlnm = dndm * M
            QHII = np.trapz(dndlnm * V, x=np.log(M))
            
            return QHII
        else:
            raise NotImplemented('Uncrecognized option for BSD.')
        
    def BubbleDensity(self, z, R=None, popid=0):
        """
        Compute the volume density of bubbles at redshift z of given radius.
        """
                
        # Can separate size and density artificially
        b_size = self.pf['bubble_size']
        b_dens = self.pf['bubble_density']
        
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
                halos = self.pops[0].halos
                
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

                logMmin = np.log10(self.Mmin(z))
                n = LinearNDInterpolator([halos.z, halos.logM], halos.ngtm)

                return n([z, logMmin])

        raise ValueError('Somethings not right')
        
    @property
    def Mmin(self):
        if not hasattr(self, '_Mmin'):
            Mmin_tab = np.ones_like(self.pops[0].halos.z) * np.inf
            for pop in self.pops:
                if not pop.is_src_ion_fl:
                    continue
                
                try:
                    Mmin_tab = np.minimum(Mmin_tab, pop._tab_Mmin)
                except AttributeError:
                    Mmin_tab = np.minimum(Mmin_tab, 10**pop.halos.logM_min)
            
            self._Mmin = lambda z: np.interp(z, self.pops[0].halos.z, Mmin_tab)
        
        return self._Mmin
        
    def _K(self, zeta):
        return erfinv(1. - 1. / zeta)
    
    def _delta_c(self, z, popid=0):
        pop = self.pops[popid]
        return pop.cosm.delta_c0 / pop.growth_factor(z)
        
    def _B0(self, z, zeta=40.):
        
        pop = self.pops[0]
        
        s = pop.halos.sigma_0
        Mmin = self.Mmin(z)
        
        sigma_min = np.interp(Mmin * zeta, pop.halos.M, s)
        return self._delta_c(z) - np.sqrt(2.) * self._K(zeta) * sigma_min
    
    def _B1(self, z, zeta=40):
        pop = self.pops[0]
        s = pop.halos.sigma_0
        sigma_min = np.interp(self.Mmin(z) * zeta, pop.halos.M, s)
        ddx_ds2 = self._K(zeta) / np.sqrt(2. * (sigma_min**2 - s**2))
    
        return ddx_ds2[s == s.min()]
    
    def _B(self, z, zeta, zeta_min):
        """
        Linear barrier.
        """
        pop = self.pops[0]
        s = pop.halos.sigma_0

        return self._B0(z, zeta_min) + self._B1(z, zeta) * s**2

    def BubbleSizeDistribution(self, z, zeta):

        if not hasattr(self, '_bsd_cache'):
            self._bsd_cache = {}

        if z in self._bsd_cache:
            return self._bsd_cache[z]

        if self.pf['bubble_size_dist'] is None:
            if self.pf['bubble_density'] is not None:
                Rb = self.pf['bubble_size']
                Mb = (4. * np.pi * Rb**3 / 3.) * self.cosm.mean_density0 \
                    / g_per_msun
                
                self._bsd_cache[z] = Rb, Mb, self.pf['bubble_density']

            else:
                raise NotImplementedError('help')
        elif self.pf['bubble_size_dist'].lower() == 'fzh04':
            Mb = self.pops[0].halos.M * zeta
            rho0 = self.cosm.mean_density0
            sig = self.pops[0].halos.sigma_0
            S = sig**2
            
            Mmin = self.Mmin(z)
            if type(zeta) == np.ndarray:
                zeta_min = np.interp(Mmin, self.pops[0].halos.M, zeta)
            else:
                zeta_min = zeta
            
            # Shouldn't there be a correction factor here to account for the
            # fact that some of the mass is He?
            
            pcross = self._B0(z, zeta_min) / np.sqrt(2. * np.pi * S**3) \
                * np.exp(-0.5 * self._B(z, zeta, zeta_min)**2 / S)
                
            R = ((Mb / rho0) * 0.75 / np.pi)**(1./3.)
            
            dndm = rho0 * pcross * 2 * np.abs(self.pops[0].halos.dlns_dlnm) \
                * S / Mb**2

            self._bsd_cache[z] = R, Mb, dndm
        else:
            raise NotImplementedError('Unrecognized option: %s' % self.pf['bubble_size_dist'])

        return self._bsd_cache[z]

    @property
    def halos(self):
        if not hasattr(self, '_halos'):
            self._halos = self.pops[0].halos
        return self._halos

    def JointProbability(self, z, dr, zeta, Tprof=None, term='ii', data=None,
        zeta_lya=None):
        """
        Compute the joint probability that two points are ionized, heated, etc.
        
        Parameters
        ----------
        z : int, float
        dr : np.ndarray
            Array of scales to consider.
        zeta : ionization parameter
        Tprof : 
        term : str
            
        """
            
        if self.pf['bubble_size_dist'].lower() == 'fzh04':
            
            Rb, Mb, dndm = self.BubbleSizeDistribution(z, zeta)
            Vb = 4. * np.pi * Rb**3 / 3.

            # More descriptive subscripts for Vsh
            if 'h' in term:
                if self.pf['bubble_shell_size_rel'] is not None:
                    Rh = Rb * (1. + self.pf['bubble_shell_size_rel'])
                else:
                    Rh = Rb + self.pf['bubble_shell_size_abs']
                    
                Vsh = 4. * np.pi * (Rh - Rb)**3 / 3.
                
            if 'a' in term:
                Ma = Mb * (zeta_lya / zeta)
                Ra = ((Ma / self.cosm.mean_density0) * 0.75 / np.pi)**(1./3.)
                Vsh = 4. * np.pi * (Ra - Rb)**3 / 3.    
            
            Mmin = self.Mmin(z)
            
            # Should tighten this up. Well, will Mmin ever NOT be in the grid?
            iM = np.argmin(np.abs(self.pops[0].halos.M - Mmin))
            
            AA = np.zeros_like(dr)
            for i, sep in enumerate(dr):
                Vo_sph = self.overlap_region_sphere(sep, Rb)
                
                if term == 'ii':
                    integrand1 = dndm[iM:] * Vo_sph[iM:]
                    integrand2 = dndm[iM:] * (Vb[iM:] - Vo_sph[iM:])
                elif term == 'hh':
                    Vo_sh_r1, Vo_sh_r2 = self.overlap_region_shell(sep, Rb, Rh-Rb)
                    Vo_hh = Vo_sh_r1 - 2. * Vo_sh_r2
                    Vo_tot = self.overlap_region_sphere(sep, Rb+Rh)
                    integrand1 = dndm[iM:] * Vo_hh[iM:]
                    integrand2 = 0.0
                    #integrand2 = dndm[iM:] * (Vsh[iM:] - Vo_tot[iM:])
                elif term == 'aa':
                    Vo_sh_r1, Vo_sh_r2 = self.overlap_region_shell(sep, Rb, Ra-Rb)
                    Vo_aa = Vo_sh_r1 - 2. * Vo_sh_r2
                    Vo_tot = self.overlap_region_sphere(sep, Rb+Ra)
                    integrand1 = dndm[iM:] * Vo_aa[iM:]
                    integrand2 = 0.0
                    #integrand2 = dndm[iM:] * (Vsh[iM:] - Vo_tot[iM:])    
                elif term == 'ih':
                    Vo_sh_r1, Vo_sh_r2 = self.overlap_region_shell(sep, Rb, Rh-Rb)
                    Vo_hi = 2 * Vo_sh_r2
                    Vo_tot = self.overlap_region_sphere(sep, Rh)
                    integrand1 = dndm[iM:] * Vo_hi[iM:]
                    integrand2 = 0.0
                elif term == 'ia':
                    Vo_sh_r1, Vo_sh_r2 = self.overlap_region_shell(sep, Rb, Ra-Rb)
                    Vo_ai = 2 * Vo_sh_r2
                    Vo_tot = self.overlap_region_sphere(sep, Ra)
                    integrand1 = dndm[iM:] * Vo_ai[iM:]
                    integrand2 = 0.0    
                elif term == 'id':
                    
                    #b = #self.halos.bias(z, np.log10(Mmin)).squeeze()
                    iz = np.argmin(np.abs(z - self.halos.z))
                    b = self.halos.bias_tab[iz]
                    dndm = self.halos.dndm[iz]
                    Mh = self.halos.M
                    
                    rho0 = self.cosm.mean_density0
                    
                    dndlnm = dndm * Mh
                    
                    corr_t1 = np.trapz(dndlnm * b, x=np.log(Mh))
                    
                    xi_dd = data['cf_dd']
                    
                    
                    
                    corr_t2 = 1e-5#np.trapz(xi_dd, x=data['dr'])
                    
                    corr = corr_t1 * corr_t2
                    
                    #import matplotlib.pyplot as pl
                    #
                    #print z, corr_t1, corr_t2, corr
                    #pl.loglog(self.halos.M, b)
                    #
                    #pl.loglog(data['dr'], np.abs(xi_dd))
                    #
                    #raw_input('<enter>')
                    
                    
                    
                    
                    
                    # Joint probability of ionization and density fields
                    # (simplified as in FZH04 (their Eq. 24)
                    integrand = dndlnm * np.exp(-b * corr) / rho0
                    integral = np.trapz(integrand, x=np.log(Mh))
                    
                    #print z, corr_t1, integral
                    
                    return (1. - data['xibar']) * (1. - integral)
                else:
                    raise NotImplementedError('help!')
                    
                exp_int1 = np.exp(-np.trapz(integrand1 * Mb[iM:], 
                    x=np.log(Mb[iM:])))                    
                
                # One halo term
                AA[i] += (1. - exp_int1) 
                
                # Two halo term
                if self.pf['include_bias']:
                    ep = self.excess_probability(z, sep, data)
                    integrand2 *= (1. + ep)
                
                exp_int2 = np.exp(-np.trapz(integrand2 * Mb[iM:], 
                    x=np.log(Mb[iM:])))
                    
                AA[i] += exp_int1 * (1. - exp_int2)**2

            return AA
            
        ##
        # Phenomenological from here down.
        ##    
        #if self.pf['bubble_size_dist'] is None:
        #    R = self.pf['bubble_size']
        #    Rs = self.pf['bubble_shell_size']
        #
        #    V = 4. * np.pi * (R+Rs)**3 / 3.
        #    V_o = self.overlap_region_shell(dr, R, Rs)
        #
        #    # Abundance of halos
        #    n_b = self.BubbleDensity(z)
        #
        #    # One and two halo terms, respectively
        #    if pop.pf['pop_one_halo_term']:
        #        oht = (1. - np.exp(-n_b * V_o))
        #    else:
        #        oht = 0.0
        #
        #    if pop.pf['pop_two_halo_term']:
        #        tht = np.exp(-n_b * V_o) * (1. - np.exp(-n_b * (V - V_o)))**2
        #    else:
        #        tht = 0.0
        #
        #    return oht + tht
    
    def excess_probability(self, z, r, data):
        pop = self.pops[0] # any one will do
        
        iz = np.argmin(np.abs(z - pop.halos.z))
        #Mmin = pop._tab_Mmin[iz]
        Mmin = self.Mmin(z)
        iM = np.argmin(np.abs(pop.halos.M - Mmin))
        
        #b = pop.halos.bias(z, pop.halos.logM[iM:]).squeeze()
        bbar = np.trapz(b)
        
        xi_dd = np.interp(r, data['dr'], data['cf_dd'].real)        
        
        return b * bbar * np.array(xi_dd)

    def CorrelationFunction(self, z, field_1, field_2, dr=None, popid=0):

        # Ionization auto-correlation function
        if field_1 == field_2 == 'x':
            #Qi  = self.BubbleFillingFactor(z, )
            Pii = self.IonizationProbability(z, dr, popid)
            return Pii #- Qi**2
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
            
        