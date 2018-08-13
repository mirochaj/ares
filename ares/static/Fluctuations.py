"""

FluctuatingBackground.py

Author: Jordan M_brocha
Affiliation: UCLA
Created on: Mon Oct 10 14:29:54 PDT 2016

Description: 

"""

import numpy as np
from ..physics import Cosmology
from ..util import ParameterFile
from scipy.special import erfinv
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from scipy.integrate import quad, simps
from ..physics.Hydrogen import Hydrogen
from ..physics.HaloModel import HaloModel
from ..util.Math import LinearNDInterpolator
from ..populations.Composite import CompositePopulation
from ..physics.CrossSections import PhotoIonizationCrossSection
from ..physics.Constants import g_per_msun, cm_per_mpc, dnu, s_per_yr, c, \
    s_per_myr, erg_per_ev, k_B, m_p, dnu

root2 = np.sqrt(2.)
four_pi = 4. * np.pi

class Fluctuations(object):
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
    def hydr(self):
        if not hasattr(self, '_hydr'):
            if self.grid is None:
                self._hydr = Hydrogen(**self.pf)
            else:
                self._hydr = self.grid.hydr
                    
        return self._hydr

    def _overlap_region(self, dr, R1, R2):
        """
        Volume of intersection between two spheres of radii R1 < R2.
        """
        
        Vo = np.pi * (R2 + R1 - dr)**2 \
            * (dr**2 + 2. * dr * R1 - 3. * R1**2 \
             + 2. * dr * R2 + 6. * R1 * R2 - 3. * R2**2) / 12. / dr
        
        if type(Vo) == np.ndarray:
            # Small-scale vs. large Scale
            SS = dr <= R2 - R1
            LS = dr >= R1 + R2
            Vo[LS == 1] = 0.0
            
            #if np.any(SS):
            if type(R1) == np.ndarray:
                Vo[SS == 1] = 4. * np.pi * R1[SS == 1]**3 / 3.
            else:
                Vo[SS == 1] = 4. * np.pi * R1**3 / 3.
            
        return Vo
        
    def IV(self, dr, R1, R2):
        """
        Just a vectorized version of the overlap calculation.
        """
        return self._overlap_region(dr, R1, R2)

    def intersectional_volumes(self, dr, R1, R2, R3):
         IV = self.IV
         
         V11 = IV(dr, R1, R1)
         
         zeros = np.zeros_like(V11)
         
         if np.all(R2 == 0):             
             return V11, zeros, zeros, zeros, zeros, zeros
             
         V12 = IV(dr, R1, R2)
         V22 = IV(dr, R2, R2)
         
         if np.all(R3 == 0):
             return V11, V12, V22, zeros, zeros, zeros
             
         V13 = IV(dr, R1, R3)
         V23 = IV(dr, R2, R3)
         V33 = IV(dr, R3, R3)
         
         return V11, V12, V22, V13, V23, V33
    
    def overlap_volumes(self, dr, R1, R2, R3):
        """
        Overlap volumes, i.e., volumes in which a source affects two points
        in different ways. For example, V11 is the volume in which a source
        ionizes both points (at separation `dr`), V12 is the volume in which
        a source ionizes one point and heats the other, and so on.
    
        In this order: V11, V12, V13, V22, V23, V33
        """
        
        IV = self.IV
    
        V11 = IV(dr, R1, R1)
        
        V12 = 2 * IV(dr, R1, R2) - IV(dr, R1, R1)
        
        # When dr < (R3 - R2) and dr < (R2 - R1), terms cancel
        V13 = 2 * IV(dr, R1, R3) - 2 * IV(dr, R1, R2)
    
        V22 = IV(dr, R2, R2) - 2. * IV(dr, R1, R2) + IV(dr, R1, R1)
                
        V23 = 2. * IV(dr, R2, R3) - IV(dr, R2, R2) \
            - 2. * IV(dr, R1, R3) + 2. * IV(dr, R1, R2) - IV(dr, R1, R1)
        V33 = IV(dr, R3, R3) - 2. * IV(dr, R2, R3) + IV(dr, R2, R2)

        return V11, V12, V13, V22, V23, V33
    
    def exclusion_volumes(self, dr, R1, R2, R3):
        """
        Volume in which a single source only affects one 
        """
        pass
    
    def BubbleShellFillingFactor(self, z, zeta, R_s):
        """
        
        """
        if self.pf['bubble_size_dist'] is None:
            R_b = self.pf['bubble_size']
            V_b = 4. * np.pi * R_b**3 / 3.
            n_b = self.BubbleDensity(z)
        
            Qh = 1. - np.exp(-n_b * V_b)
        elif self.pf['bubble_size_dist'].lower() == 'fzh04':
            R_b, M_b, dndm_b = self.BubbleSizeDistribution(z, zeta)
            Qi = self.MeanIonizedFraction(z, zeta)
            
            if type(R_s) is np.ndarray:
                nz = R_b > 0
                assert np.allclose(np.diff(R_s[nz==1] / R_b[nz==1]), 0.0), \
                    "No support for absolute scaling of hot bubbles yet."

                Qh = Qi * ((R_s[0]**3 - R_b[0]**3) / R_b[0]**3)

                return np.minimum(Qh, 1.) #- Qi)
            else:
                return 0.0
            
            #if np.logical_and(np.all(R_s == 0), np.all(Rc == 0)):
            #    return 0.
            #
            #Mmin = self.Mmin(z)
            #iM = np.argmin(np.abs(Mmin * zeta - M_b))
            #
            #Vi = 4. * np.pi * R_b**3 / 3.   
            #Vsh1 = 4. * np.pi * (R_s**3 - R_b**3) / 3.
            #
            #dndlnm = dndm * M_b
            #
            #if np.all(Rc == 0):
            #    Vsh2 = Qc = 0.0
            #else:
            #    Vsh2 = 4. * np.pi * (Rc - R_s)**3 / 3.
            #    Qc = np.trapz(dndlnm[iM:] * Vsh2[iM:], x=np.log(M_b[iM:]))
            #
            #Qi = np.trapz(dndlnm[iM:] * Vi[iM:], x=np.log(M_b[iM:]))
            #Qh = np.trapz(dndlnm[iM:] * Vsh1[iM:], x=np.log(M_b[iM:]))
            
            #if self.pf['powspec_rescale_Qion'] and self.pf['powspec_rescale_Qhot']:
            #    norm = min(zeta * self.halos.fcoll_2d(z, np.log10(self.Mmin(z))), 1)
            #    
            #    corr = (norm / Qi)
            #    Qh *= corr

        else:
            raise NotImplemented('Uncrecognized option for BSD.')
         
        #return min(Qh, 1.), min(Qc, 1.)

    @property
    def bsd_model(self):
        return self.pf['bubble_size_dist'].lower()

    def MeanIonizedFraction(self, z, zeta):
        Mmin = self.Mmin(z)
        logM = np.log10(Mmin)
        
        return np.minimum(1.0, zeta * self.halos.fcoll_2d(z, logM))

    def BubbleFillingFactor(self, z, zeta, rescale=True):
        """
        Fraction of volume filled by bubbles.
        """
        

        if self.bsd_model is None:
            R_b = self.pf['bubble_size']
            Vi = 4. * np.pi * R_b**3 / 3.
            ni = self.BubbleDensity(z)

            Qi = 1. - np.exp(-ni * Vi)

        elif self.bsd_model in ['fzh04', 'hmf']:
            
            # Smallest bubble is one around smallest halo.
            # Don't actually need its mass, just need index to correctly
            # truncate integral.
            Mmin = self.Mmin(z) * zeta
            logM = np.log10(Mmin)

            # M_b should just be self.m? No.
            R_b, M_b, dndm_b = self.BubbleSizeDistribution(z, zeta, rescale)
            Vi = 4. * np.pi * R_b**3 / 3.
                        
            iM = np.argmin(np.abs(Mmin - M_b))
            
            Qi = simps(dndm_b[iM:] * M_b[iM:] * Vi[iM:], x=np.log(M_b[iM:]))
            
            # This means reionization is over.
            if self.bsd_model == 'fzh04':
                if self._B0(z, zeta) <= 0:
                    return 1.
                
        else:
            raise NotImplemented('Uncrecognized option for BSD.')
        
        return min(Qi, 1.)
        
        # Grab heated phase to enforce BC
        #Rs = self.BubbleShellRadius(z, R_b)        
        #Vsh = 4. * np.pi * (Rs - R_b)**3 / 3.
        #Qh = np.trapz(dndm * Vsh * M_b, x=np.log(M_b))   
        
        #if lya and self.pf['bubble_pod_size_func'] in [None, 'const', 'linear']:
        #    Rc = self.BubblePodRadius(z, R_b, zeta, zeta_lya)
        #    Vc = 4. * np.pi * (Rc - R_b)**3 / 3.
        #    
        #    if self.pf['powspec_rescale_Qlya']:
        #        # This isn't actually correct since we care about fluxes
        #        # not number of photons, but fine for now.
        #        Qc = min(zeta_lya * self.halos.fcoll_2d(z, np.log10(self.Mmin(z))), 1)
        #    else:
        #        Qc = np.trapz(dndlnm[iM:] * Vc[iM:], x=np.log(M_b[iM:]))
        #        
        #    return min(Qc, 1.)
        #
        #elif lya and self.pf['bubble_pod_size_func'] == 'fzh04':
        #    return self.BubbleFillingFactor(z, zeta_lya, None, lya=False)
        #else:
                
    @property
    def tab_Mmin(self):
        if not hasattr(self, '_tab_Mmin'):
            raise AttributeError('Must set Mmin by hand (right now)')
        
        return self._tab_Mmin
        
    @tab_Mmin.setter
    def tab_Mmin(self, value):
        if type(value) is not np.ndarray:
            value = np.ones_like(self.halos.tab_z) * value
        else:
            assert value.size == self.halos.tab_z.size
            
        self._tab_Mmin = value
        
    def Mmin(self, z):
        return np.interp(z, self.halos.tab_z, self.tab_Mmin)

    def mean_halo_bias(self, z):
        bias = self.halos.Bias(z)
        
        M_h = self.halos.tab_M
        iz_h = np.argmin(np.abs(z - self.halos.tab_z))
        iM_h = np.argmin(np.abs(self.Mmin(z) - M_h))
        
        dndm_h = self.halos.tab_dndm[iz_h]
        
        return 1.0
    
        #return simps(M_h * dndm_h * bias, x=np.log(M_h)) \
        #    / simps(M_h * dndm_h, x=np.log(M_h))
        
    def bubble_bias(self, z, zeta):
        """
        Eq. 9.24 in Loeb & Furlanetto (2013) or Eq. 22 in FZH04.
        """
        iz = np.argmin(np.abs(z - self.halos.tab_z))
        s = self.sigma
        S = s**2
    
        return 1. + ((self._B(z, zeta, zeta) / S - (1. / self._B0(z, zeta))) \
            / self._growth_factor(z))    

    def mean_bubble_bias(self, z, zeta, term='ii'):
        """
        """
                
        R, M_b, dndm_b = self.BubbleSizeDistribution(z, zeta)
    
        #if ('h' in term) or ('c' in term) and self.pf['powspec_temp_method'] == 'shell':
        #    R_s, Rc = self.BubbleShellRadius(z, R_b)
        #    R = R_s
        #else:
    
        V = 4. * np.pi * R**3 / 3.
    
        Mmin = self.Mmin(z) * zeta
        iM = np.argmin(np.abs(Mmin - self.m))    
        bHII = self.bubble_bias(z, zeta)
        
        #tmp = dndm[iM:]
        #print(z, len(tmp[np.isnan(tmp)]), len(bHII[np.isnan(bHII)]))
            
        #imax = int(min(np.argwhere(np.isnan(R_b))))
    
        Qi = self.MeanIonizedFraction(z, zeta)
        
        
        return simps(dndm_b[iM:] * V[iM:] * bHII[iM:] * M_b[iM:],
            x=np.log(M_b[iM:])) / Qi
    
    #def delta_bubble_mass_weighted(self, z, zeta):
    #    if self._B0(z, zeta) <= 0:
    #        return 0.
    #            
    #    R_b, M_b, dndm_b = self.BubbleSizeDistribution(z, zeta)   
    #    Vi = 4. * np.pi * R_b**3 / 3.
    #         
    #    Mmin = self.Mmin(z) * zeta
    #    iM = np.argmin(np.abs(Mmin - self.m))    
    #    B = self._B(z, zeta)
    #    rho0 = self.cosm.mean_density0
    #    
    #    dm_ddel = rho0 * Vi
    #
    #    return simps(B[iM:] * dndm_b[iM:] * M_b[iM:], x=np.log(M_b[iM:]))
    
    def delta_bubble_vol_weighted(self, z, zeta):
        if not self.pf['ps_include_ion']:
            return 0.0
        
        if not self.pf['ps_include_xcorr_ion_rho']:
            return 0.0    
            
        if self._B0(z, zeta) <= 0:
            return 0.
    
        R_b, M_b, dndm_b = self.BubbleSizeDistribution(z, zeta)   
        Vi = 4. * np.pi * R_b**3 / 3.
    
        Mmin = self.Mmin(z) * zeta
        iM = np.argmin(np.abs(Mmin - self.m))    
        B = self._B(z, zeta)
        
        return simps(B[iM:] * dndm_b[iM:] * Vi[iM:] * M_b[iM:], 
            x=np.log(M_b[iM:]))
    
   #def mean_bubble_overdensity(self, z, zeta):
   #    if self._B0(z, zeta) <= 0:
   #        return 0.
   #            
   #    R_b, M_b, dndm_b = self.BubbleSizeDistribution(z, zeta)  
   #    Vi = 4. * np.pi * R_b**3 / 3.
   #         
   #    Mmin = self.Mmin(z) * zeta
   #    iM = np.argmin(np.abs(Mmin - self.m))
   #    B = self._B(z, zeta)
   #    rho0 = self.cosm.mean_density0
   #    
   #    dm_ddel = rho0 * Vi
   #
   #    return simps(B[iM:] * dndm_b[iM:] * M_b[iM:], x=np.log(M_b[iM:]))
        
    def mean_halo_abundance(self, z, Mmin=False):
        M_h = self.halos.tab_M
        iz_h = np.argmin(np.abs(z - self.halos.tab_z))
        
        if Mmin:
            iM_h = np.argmin(np.abs(self.Mmin(z) - M_h))
        else:
            iM_h = 0
            
        dndm_h = self.halos.tab_dndm[iz_h]
    
        return simps(M_h * dndm_h, x=np.log(M_h))
        
    def spline_cf_mm(self, z):
        if not hasattr(self, '_spline_cf_mm_'):
            self._spline_cf_mm_ = {}
            
        if z not in self._spline_cf_mm_:
            iz = np.argmin(np.abs(z - self.halos.tab_z_ps))
            self._spline_cf_mm_[z] = interp1d(np.log(self.halos.tab_R), 
                self.halos.tab_cf_mm[iz], kind='cubic', bounds_error=False,
                fill_value=0.0)
        
        return self._spline_cf_mm_[z]    
    
    def excess_probability(self, z, zeta, R, term='ii'):
        """
        This is the excess probability that a point is ionized given that 
        we already know another point (at distance r) is ionized.
        """
                
        # Function of bubble mass (bubble size)
        bHII = self.bubble_bias(z, zeta)
        bbar = self.mean_bubble_bias(z, zeta, term)
        
        if R < self.halos.tab_R.min():
            print("R too small")
        if R > self.halos.tab_R.max():
            print("R too big")

        xi_dd = self.spline_cf_mm(z)(np.log(R))

        #if term == 'ii':
        return bHII * bbar * xi_dd
        #elif term == 'id':
        #    return bHII * bbar * xi_dd 
        #else:
        #    raise NotImplemented('help!')

    def _K(self, zeta):
        return erfinv(1. - (1. / zeta))
    
    def _growth_factor(self, z):
        return np.interp(z, self.halos.tab_z, self.halos.tab_growth,
            left=np.inf, right=np.inf)
    
    def _delta_c(self, z):
        return self.cosm.delta_c0 / self._growth_factor(z)

    def _B0(self, z, zeta=40.):

        iz = np.argmin(np.abs(z - self.halos.tab_z))
        s = self.sigma
        
        # Variance on scale of smallest collapsed object
        sigma_min = self.sigma_min(z, zeta)
        
        return self._delta_c(z) - root2 * self._K(zeta) * sigma_min
    
    def _B1(self, z, zeta=40.):
        iz = np.argmin(np.abs(z - self.halos.tab_z))
        s = self.sigma #* self.halos.growth_factor[iz]
                
        sigma_min = self.sigma_min(z, zeta)
        
        return self._K(zeta) / np.sqrt(2. * sigma_min**2)
    
    def _B(self, z, zeta, zeta_min=None):
        return self.LinearBarrier(z, zeta, zeta_min=None)
        
    def LinearBarrier(self, z, zeta, zeta_min=None):
        iz = np.argmin(np.abs(z - self.halos.tab_z))
        s = self.sigma #/ self.halos.growth_factor[iz]
        
        if zeta_min is None:
            zeta_min = zeta
        
        return self._B0(z, zeta_min) + self._B1(z, zeta) * s**2
    
    def Barrier(self, z, zeta, zeta_min):
        """
        Full barrier.
        """
        
        #iz = np.argmin(np.abs(z - self.halos.tab_z))
        #D = self.halos.growth_factor[iz]

        sigma_min = self.sigma_min(z, zeta)
        #Mmin = self.Mmin(z)
        #sigma_min = np.interp(Mmin, self.halos.M, self.halos.sigma_0)

        delta = self._delta_c(z)

        return delta - np.sqrt(2.) * self._K(zeta) \
            * np.sqrt(sigma_min**2 - self.sigma**2)
        
        #return self.cosm.delta_c0 - np.sqrt(2.) * self._K(zeta) \
        #    * np.sqrt(sigma_min**2 - s**2)

    def sigma_min(self, z, zeta):
        Mmin = self.Mmin(z)
        return np.interp(Mmin, self.halos.tab_M, self.halos.tab_sigma)

    #def BubblePodSizeDistribution(self, z, zeta):
    #    if self.pf['powspec_lya_method'] == 1:
    #        # Need to modify zeta and critical threshold
    #        Rc, Mc, dndm = self.BubbleSizeDistribution(z, zeta)
    #        return Rc, Mc, dndm
    #    else:
    #        raise NotImplemented('help please')

    @property
    def m(self):
        """
        Mass array used for bubbles.
        """
        if not hasattr(self, '_m'):
            self._m = 10**np.arange(5, 18.01, 0.01)
        return self._m

    @property
    def sigma(self):
        if not hasattr(self, '_sigma'):
            self._sigma = np.interp(self.m, self.halos.tab_M, self.halos.tab_sigma)
            
            # Crude but chill it's temporary
            bigm = self.m > self.halos.tab_M.max()
            if np.any(bigm):
                print("WARNING: Extrapolating sigma to higher masses.")
            
                slope = np.diff(np.log10(self.halos.tab_sigma[-2:])) \
                      / np.diff(np.log10(self.halos.tab_M[-2:]))
                self._sigma[bigm == 1] = self.halos.tab_sigma[-1] \
                    * (self.m[bigm == 1] / self.halos.tab_M.max())**slope
            
        return self._sigma
        
    @property
    def dlns_dlnm(self):
        if not hasattr(self, '_dlns_dlnm'):
            self._dlns_dlnm = np.interp(self.m, self.halos.tab_M, self.halos.tab_dlnsdlnm)
        
            bigm = self.m > self.halos.tab_M.max()
            if np.any(bigm):
                print("WARNING: Extrapolating dlns_dlnm to higher masses.")
                slope = np.diff(np.log10(np.abs(self.halos.tab_dlnsdlnm[-2:]))) \
                      / np.diff(np.log10(self.halos.tab_M[-2:]))
                self._dlns_dlnm[bigm == 1] = self.halos.tab_dlnsdlnm[-1] \
                    * (self.m[bigm == 1] / self.halos.tab_M.max())**slope
        
        return self._dlns_dlnm

    def BubbleSizeDistribution(self, z, zeta, rescale=True):
        """
        Compute the ionized bubble size distribution.
        
        Parameters
        ----------
        z: int, float
            Redshift of interest.
        zeta : int, float, np.ndarray
            Ionizing efficiency.
            
        Returns
        -------
        Tuple containing (in order) the bubble radii, masses, and the
        differential bubble size distribution. Each is an array of length
        self.halos.tab_M, i.e., with elements corresponding to the masses
        used to compute the variance of the density field.
            
        """

        reionization_over = False 
           
        # Comoving matter density
        rho0_m = self.cosm.mean_density0
        rho0_b = rho0_m * self.cosm.fbaryon 
        
        # Mean (over-)density of bubble material
        delta_B = self._B(z, zeta, zeta)
           
        if self.bsd_model is None:
            if self.pf['bubble_density'] is not None:
                R_b = self.pf['bubble_size']
                M_b = (4. * np.pi * Rb**3 / 3.) * rho0_m
                dndm = self.pf['bubble_density']
            else:
                raise NotImplementedError('help')
        
        elif self.bsd_model == 'hmf':
            M_b = self.halos.tab_M * zeta
            # Assumes bubble material is at cosmic mean density
            R_b = (3. * M_b / rho0_b / 4. / np.pi)**(1./3.)
            iz = np.argmin(np.abs(z - self.halos.tab_z))
            dndm = self.halos.tab_dndm[iz].copy()
        
        elif self.bsd_model == 'fzh04':
            # Just use array of halo mass as array of ionized region masses.
            # Arbitrary at this point, just need an array of masses.
            # Plus, this way, the sigma's from the HMF are OK.
            M_b = self.m 
                                    
            # Radius of ionized regions as function of delta (mass)
            R_b = (3. * M_b / rho0_m / (1. + delta_B) / 4. / np.pi)**(1./3.)
        
            Vi = four_pi * R_b**3 / 3.
            
            # This is Eq. 9.38 from Steve's book.
            # The factors of 2, S, and M_b are from using dlns instead of 
            # dS (where S=s^2)
            dndm = rho0_m * self.pcross(z, zeta) * 2 * np.abs(self.dlns_dlnm) \
                * self.sigma**2 / M_b**2
                
            # Reionization is over!
            # Only use barrier condition if we haven't asked to rescale
            # or supplied Q ourselves.
            if self._B0(z, zeta) <= 0:
                reionization_over = True
                dndm = np.zeros_like(dndm)
            #elif Q is not None:
            #    if Q == 1:
            #        reionization_over = True
            #        dndm = np.zeros_like(dndm)

        else:
            raise NotImplementedError('Unrecognized option: %s' % self.pf['bubble_size_dist'])
            
        
        # This is a trick to guarantee that the integral over the bubble
        # size distribution yields the mean ionized fraction.    
        if (not reionization_over) and rescale:
            Mmin = self.Mmin(z) * zeta
            iM = np.argmin(np.abs(M_b - Mmin))
            Qi = simps(dndm[iM:] * Vi[iM:] * M_b[iM:], x=np.log(M_b[iM:]))
            xibar = self.MeanIonizedFraction(z, zeta)
            dndm *= -np.log(1. - xibar) / Qi
            
        return R_b, M_b, dndm
        
    def pcross(self, z, zeta):
        """
        Up-crossing probability.
        """
        
        S = self.sigma**2    
        Mmin = self.Mmin(z) #* zeta # doesn't matter for zeta=const
        if type(zeta) == np.ndarray:
            raise NotImplemented('this is wrong.')
            zeta_min = np.interp(Mmin, self.m, zeta)
        else:
            zeta_min = zeta
        
        zeros = np.zeros_like(self.sigma)
            
        B0 = self._B0(z, zeta_min)
        B1 = self._B1(z, zeta)
        Bl = self.LinearBarrier(z, zeta, zeta_min)
        p = (B0 / np.sqrt(2. * np.pi * S**3)) * np.exp(-0.5 * Bl**2 / S)
        
        #p = (B0 / np.sqrt(2. * np.pi * S**3)) \
        #    * np.exp(-0.5 * B0**2 / S) * np.exp(-B0 * B1) * np.exp(-0.5 * B1**2 / S)
        
        return p#np.maximum(p, zeros)
        
    @property
    def halos(self):
        if not hasattr(self, '_halos'):
            self._halos = HaloModel(**self.pf)
        return self._halos
        
    @property
    def Emin_X(self):
        if not hasattr(self, '_Emin_X'):
            xrpop = None
            for i, pop in enumerate(self.pops):
                if not pop.is_src_heat_fl:
                    continue
                    
                if xrpop is not None:
                    raise AttributeError('help! can only handle 1 X-ray pop right now')
                    
                xrpop = pop
                
            self._Emin_X = pop.src.Emin
            
        return self._Emin_X

    def get_Nion(self, z, R_b):
        return 4. * np.pi * (R_b * cm_per_mpc / (1. + z))**3 \
            * self.cosm.nH(z) / 3.
                    
    def _cache_jp(self, z, term):
        if not hasattr(self, '_cache_jp_'):
            self._cache_jp_ = {}
        
        if z not in self._cache_jp_:
            self._cache_jp_[z] = {}
        
        if term not in self._cache_jp_[z]:
            return None
        else:
            #print("Loaded P_{} at z={} from cache.".format(term, z))
            return self._cache_jp_[z][term]
        
    def _cache_cf(self, z, term):
        if not hasattr(self, '_cache_cf_'):
            self._cache_cf_ = {}
    
        if z not in self._cache_cf_:
            self._cache_cf_[z] = {}
    
        if term not in self._cache_cf_[z]:
            return None
        else:
            #print("Loaded xi_{} at z={} from cache.".format(term, z))
            return self._cache_cf_[z][term]    
    
    def _cache_ps(self, z, term):
        if not hasattr(self, '_cache_ps_'):
            self._cache_ps_ = {}
    
        if z not in self._cache_ps_:
            self._cache_ps_[z] = {}
    
        if term not in self._cache_ps_[z]:
            return None
        else:
            return self._cache_ps_[z][term]        
    
    @property
    def is_Rs_const(self):
        if not hasattr(self, '_is_Rs_const'):
            self._is_Rs_const = False
        return self._is_Rs_const
        
    @is_Rs_const.setter
    def is_Rs_const(self, value):
        self._is_Rs_const = value
    
    def _cache_Vo(self, z):
        if not hasattr(self, '_cache_Vo_'):
            self._cache_Vo_ = {}

        if z in self._cache_Vo_:
            return self._cache_Vo_[z]
        
        if self.is_Rs_const and len(self._cache_Vo_.keys()) > 0:
            return self._cache_Vo_[self._cache_Vo_.keys()[0]]

        return None
        
    def _cache_IV(self, z):
        if not hasattr(self, '_cache_IV_'):
            self._cache_IV_ = {}
    
        if z in self._cache_IV_:
            return self._cache_IV_[z]
            
        if self.is_Rs_const and len(self._cache_IV_.keys()) > 0:
            return self._cache_IV_[self._cache_IV_.keys()[0]]    
    
        return None    
        
    def _cache_p(self, z, term):
        if not hasattr(self, '_cache_p_'):
            self._cache_p_ = {}
    
        if z not in self._cache_p_:
            self._cache_p_[z] = {}
    
        if term not in self._cache_p_[z]:
            return None
        else:
            return self._cache_p_[z][term]
    
    def mean_halo_overdensity(self, z):
        # Mean density of halos (mass is arbitrary)
        rho_h = self.halos.MeanDensity(1e8, z) * cm_per_mpc**3 / g_per_msun
        return rho_h / self.cosm.mean_density0 - 1.
        
    def fcoll_vol(self, z, Mmin=0.0, Mmax=None):
        """
        This may not be quite right, since we just integrate over the mass
        range we have....
        """
        M_h = self.halos.tab_M
        iz_h = np.argmin(np.abs(z - self.halos.tab_z))
        dndm_h = self.halos.tab_dndm[iz_h]
                                
        # Volume of halos (within virial radii)
        Rvir = self.halos.VirialRadius(M_h, z) / 1e3 # Convert to Mpc
        Vvir = 4. * np.pi * Rvir**3 / 3.
        
        return self.get_prob(z, M_h, dndm_h, Mmin, Vvir, exp=False, ep=0.0, Mmax=Mmax)
    
    def ExpectationValue1pt(self, z, zeta, term='i', R_s=None, R3=None, 
        Th=500.0, Ts=None, Tk=None, Ja=None):
        """
        Compute the probability that a point is something.
        
        These are the one point terms in brackets, e.g., <x>, <x delta>, etc.
        
        Note that use of the asterisk is to imply that both quatities
        are in the same set of brackets. Maybe a better way to handle this
        notationally...
        
        """
        
        ##
        # Check cache for match
        ##
        cached_result = self._cache_p(z, term)
        
        if cached_result is not None:
            return cached_result
        
        ##
        # Otherwise, get to it.
        ##
        if term == 'i':
            val = self.MeanIonizedFraction(z, zeta)
        elif term == 'n':
            val = 1. - self.ExpectationValue1pt(z, zeta, term='i', 
                R_s=R_s, Th=Th, Ts=Ts)
        elif term == 'h':
            assert R_s is not None
            val = self.BubbleShellFillingFactor(z, zeta, R_s) 
        elif term in ['m', 'd']:
            val = 0.0
        elif term in ['n*d', 'i*d']:
            if self.pf['ps_include_xcorr_ion_rho']: 
                Qi = self.MeanIonizedFraction(z, zeta)
                del_i = self.delta_bubble_vol_weighted(z, zeta)

                if term == 'i*d':
                    val = Qi * del_i
                else:
                    val = -Qi * del_i
            else:
                val = 0.0
        elif term in ['id', 'nd']:
            val = 0.0
        elif term.strip() == 'i*h':
            assert R_s is not None
            Qi = self.MeanIonizedFraction(z, zeta)
            Qh = self.BubbleShellFillingFactor(z, zeta, R_s)
            val = Qh * Qi # implicit * 1
        elif term.strip() == 'n*h':
            # <xh> = <h> - <x_i h> = <h>
            c = self.TempToContrast(z, Th=Th, Ts=Ts)
            val = Qh #* c
        elif term.strip() == 'i*c':
            val = 0.0 # in binary model this is always true
            
            #if self.pf['ps_include_xcorr_ion_hot']:
            #    c = self.TempToContrast(z, Th=Th, Ts=Ts)
            #    Qi = self.MeanIonizedFraction(z, zeta)
            #    Qh = self.BubbleShellFillingFactor(z, zeta, R_s) 
            #    val = Qh * Qi * c
            ## In binary model, ionized points have c=0, c>0 points have i=False
            ##assert R_s is not None
            #else:
            #    val = 0.0
                
        elif term == 'c':
            Qh = self.BubbleShellFillingFactor(z, zeta, R_s)
            c = self.TempToContrast(z, Th=Th, Ts=Ts)
            val = c * Qh
        elif term.strip() == 'n*c':
            # <xc> = <c> - <x_i c> 
            #      = <c> in binary framework
            #if self.pf['ps_include_xcorr_ion_hot']: 
            c = self.TempToContrast(z, Th=Th, Ts=Ts)
            Qh = self.BubbleShellFillingFactor(z, zeta, R_s)
            #avg_x = self.ExpectationValue(z, zeta, term='n')
            val = c * Qh
            #else:
            #val = 0.0
        elif term.strip() == 'n*d*c':
            # <xdc> = <dc> - <x_i d c>
            if self.pf['ps_include_xcorr_ion_rho'] and self.pf['ps_include_xcorr_ion_hot']:                
                print("assuming <xdc>=0")
                val = 0.0
            else:
                val = 0.0
        elif term == 'psi':
            # <psi> = <x (1 + d)> = <x> + <xd> = 1 - <x_i> + <d> - <x_i d>
            #       = 1 - <x_i> - <x_i d>

            avg_xd = self.ExpectationValue1pt(z, zeta, term='n*d',
                R_s=R_s, Th=Th, Ts=Ts)
            avg_x = self.ExpectationValue1pt(z, zeta, term='n',
                R_s=R_s, Th=Th, Ts=Ts)

            val = avg_x + avg_xd

        elif term in ['phi']:
            # <phi> = <psi * (1 + c)> = <psi> + <psi * c>
            #
            # <psi * c> = <x * c> + <x * c * d>
            #           = <c> - <x_i c> + <x * c * d>
            #           = <c> - 0 + <cd> - <x_i c * d>
            #           = <c> if binary model and no density xcorr
            
            avg_xcd = self.ExpectationValue1pt(z, zeta, term='n*d*c',
                R_s=R_s, Th=Th, Ts=Ts)
                
            # Equivalent to <c> in binary field model.            
            avg_xc = self.ExpectationValue1pt(z, zeta, term='n*c',
                R_s=R_s, Th=Th, Ts=Ts)
            
            avg_psi = self.ExpectationValue1pt(z, zeta, term='psi',
                R_s=R_s, Th=Th, Ts=Ts)
            avg_psi_c = avg_xc + avg_xcd
            
            val = avg_psi + avg_psi_c
                    
        else:
            raise ValueError('Don\' know how to handle <{}>'.format(term))
        
        self._cache_p_[z][term] = val
        
        return val
    
    def ExpectationValue2pt(self, z, zeta, R, term='ii', R_s=None, R3=None, 
        Th=500.0, Ts=None, Tk=None, Ja=None, k=None):
        """
        Essentially a wrapper around JointProbability that scales
        terms like <cc'>, <xc'>, etc., from their component probabilities
        <hh'>, <ih'>, etc.
        
        Parameters
        ----------
        z : int, float
        zeta : int, float
            Ionization efficiency
        R : np.ndarray
            Array of scales to consider.
        term : str
        
        Returns
        -------
        Tuple: total, one-source, two-source contributions to joint probability.
        
        
        """
        
        ##
        # Check cache for match
        ##
        cached_result = self._cache_jp(z, term)
        
        if cached_result is not None:
            _R, _jp, _jp1, _jp2 = cached_result
            
            if _R.size == R.size:
                if np.allclose(_R, R):
                    return cached_result[1:]
            
            print("interpolating jp_{}".format(ii))
            return np.interp(R, _R, _jp), np.interp(R, _R, _jp1), np.interp(R, _R, _jp2)
        
        # Remember, we scaled the BSD so that these two things are equal
        # by construction.
        xibar = Q = Qi = self.MeanIonizedFraction(z, zeta)
        
        Rones  = np.ones_like(R)
        Rzeros = np.zeros_like(R)
        
        # If reionization is over, don't waste our time!
        if xibar == 1:
            return np.ones(R.size), Rzeros, Rzeros
        
        iz = np.argmin(np.abs(z - self.halos.tab_z_ps))
        iz_hmf = np.argmin(np.abs(z - self.halos.tab_z))
        
        # Grab the matter power spectrum
        if R.size == self.halos.tab_R.size:
            if np.allclose(R, self.halos.tab_R):
                xi_dd = self.halos.tab_cf_mm[iz]
            else:
                xi_dd = self.spline_cf_mm(z)(np.log(R))
        else:
            xi_dd = self.spline_cf_mm(z)(np.log(R))
            
        ## 
        # Before we begin: anything we're turning off?
        ##
        if not self.pf['ps_include_ion']:
            if term == 'ii':
                self._cache_jp_[z][term] = R, Qi**2 * Rones, Rzeros, Rzeros
                return Qi**2 * Rones, Rzeros, Rzeros
            elif term in ['id']:
                self._cache_jp_[z][term] = R, Rzeros, Rzeros, Rzeros
                return Rzeros, Rzeros, Rzeros
            elif term == 'idd':
                ev2pt = Qi * xi_dd
                self._cache_jp_[z][term] = R, ev2pt, Rzeros, Rzeros
                return ev2pt, Rzeros, Rzeros     # 
            elif term == 'iidd':
                ev2pt = Qi**2 * xi_dd
                self._cache_jp_[z][term] = R, ev2pt, Rzeros, Rzeros
                return ev2pt, Rzeros, Rzeros
                
            # also iid, iidd
                            
        ##
        # Check for derived quantities like psi, phi
        ##
        if term == 'psi':
            # <psi psi'> = <x (1 + d) x' (1 + d')> = <xx'(1+d)(1+d')>
            #            = <xx'(1 + d + d' + dd')>
            #            = <xx'> + 2<xx'd> + <xx'dd'>
        
            xx, xx1, xx2 = self.ExpectationValue2pt(z, zeta, R=R, term='nn',
                R_s=R_s, R3=R3, Th=Th, Ts=Ts)
        
            xxd, xxd1, xxd2 = self.ExpectationValue2pt(z, zeta, R=R, term='nnd',
                R_s=R_s, R3=R3, Th=Th, Ts=Ts)
        
            xxdd, xxdd1, xxdd2 = self.ExpectationValue2pt(z, zeta, R=R, 
                term='xxdd', R_s=R_s, R3=R3, Th=Th, Ts=Ts)
        
            ev2pt = xx + 2. * xxd + xxdd
            
            ev2pt_1 = np.zeros_like(xx1)
            ev2pt_2 = np.zeros_like(xx)
            
            self._cache_jp_[z][term] = R, ev2pt, ev2pt_1, ev2pt_2
        
            return ev2pt, ev2pt_1, ev2pt_2
        
        elif term in ['phi', '21']:
            Phi, junk1, junk2 = self.ExpectationValue2pt(z, zeta, R, term='Phi',
                R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja, k=k)
                            
            ev_psi, ev_psi1, ev_psi2 = self.ExpectationValue2pt(z, zeta, R,
                term='psi', R_s=R_s, R3=R3, Th=Th, Ts=Ts)
            #avg_psi = self.ExpectationValue1pt(z, zeta, term='psi',
            #    R_s=R_s, R3=R3, Th=Th, Ts=Ts)
            #avg_phi = self.ExpectationValue1pt(z, zeta, term='phi', 
            #    R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)
                    
            ev2pt = ev_psi + Phi
            
            self._cache_jp_[z][term] = R, ev2pt, Rzeros, Rzeros
            
            return ev2pt, Rzeros, Rzeros
        
        elif term == 'Phi':
            
            if not (self.pf['ps_include_temp'] or self.pf['ps_include_lya']):
                return Rzeros, Rzeros, Rzeros
            
            #avg_c = self.ExpectationValue1pt(z, zeta, term='c',
            #    R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)

            ev_xxc, ev_xxc1, ev_xxc2 = \
                self.ExpectationValue2pt(z, zeta, R, term='xxc',
                R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja, k=k)
            ev_xxcc, ev_xxcc1, ev_xxcc2 = \
                self.ExpectationValue2pt(z, zeta, R, term='xxcc',
                R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja, k=k)
        
            Phi = 2. * ev_xxc + ev_xxcc
        
            return Phi, Rzeros, Rzeros        
        
        elif term == 'cc':
            
            result = Rzeros.copy()
            if self.pf['ps_include_temp']:
            
                jp_hh, jp_hh1, jp_hh2 = \
                    self.ExpectationValue2pt(z, zeta, R, term='hh',
                    R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk)
                c = self.TempToContrast(z, Th=Th, Ts=Ts)
                
                result += jp_hh * c**2
                
            if self.pf['ps_include_lya']:
                ev_aa, ev_aa1, ev_aa2 = \
                    self.ExpectationValue2pt(z, zeta, R, term='aa',
                    R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, k=k)
                xa = self.hydr.RadiativeCouplingCoefficient(z, Ja, Tk)
                
                result += ev_aa / (1. + xa)**2
                
            return result, Rzeros, Rzeros
        
        elif term == 'ic':
            
            #c = self.TempToContrast(z, Th=Th, Ts=Ts)
            avg_c = self.ExpectationValue1pt(z, zeta, term='c', 
                R_s=R_s, R3=R3, Th=Th, Ts=Ts)
            
            if not self.pf['ps_include_xcorr_ion_hot']:
                return Qi * avg_c * Rones, Rzeros, Rzeros
            
            jp_ih, jp_ih1, jp_ih2 = \
                self.ExpectationValue2pt(z, zeta, R, term='ih',
                R_s=R_s, R3=R3, Th=Th, Ts=Ts)
        
            c = self.TempToContrast(z, Th=Th, Ts=Ts)
            return jp_ih * c, jp_ih1 * c, jp_ih2 * c
        
        elif term in ['xxc', 'nnc']:
            
            # Even without correlations in ionization and temperature,
            # this term is non-zero
            if not self.pf['ps_include_3pt']:
                return Rzeros, Rzeros, Rzeros
            
            # <xx'c> = <c(1 - x_i - x_i' + x_i x_i')>
            #        = <c> - <x_i c> - <x_i c'> + <x_i x_i' c>
            #        = <c> - <x_i c'> in binary model
            avg_c = self.ExpectationValue1pt(z, zeta, term='c', 
                R_s=R_s, R3=R3, Th=Th, Ts=Ts)
            avg_ic = self.ExpectationValue1pt(z, zeta, term='i*c', 
                R_s=R_s, R3=R3, Th=Th, Ts=Ts)
            ev_ic, ev_ic1, ev_ic2 = \
                self.ExpectationValue2pt(z, zeta, R, term='ic',
                R_s=R_s, R3=R3, Th=Th, Ts=Ts)
            #ev_iic, ev_iic1, ev_iic2 = \
            #    self.ExpectationValue2pt(z, zeta, R, term='iic',
            #    R_s=R_s, R3=R3, Th=Th, Ts=Ts)    
            ev_iic = 0.0
            
            return avg_c - avg_ic - ev_ic + ev_iic, Rzeros, Rzeros
            #return avg_xc - ev_ic, Rzeros, Rzeros
        
        elif term == 'xxcc':
            if not self.pf['ps_include_4pt']:
                return Rzeros, Rzeros, Rzeros
                
            xx, xx1, xx2 = self.ExpectationValue2pt(z, zeta, R, term='nn',
                R_s=R_s, R3=R3, Th=Th, Ts=Ts)
            cc, cc1, cc2 = self.ExpectationValue2pt(z, zeta, R, term='cc',
                R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja, k=k)

            #if self.pf['ps_include_density_xcorr']:
            
            #else:
            #    xc = avg_xc = 0.0

            #if self.pf['ps_use_wick']:
            iicc, iicc1, iicc2 = self.ExpectationValue2pt(z, zeta, R, term='iicc',
                R_s=R_s, R3=R3, Th=Th, Ts=Ts)
            icc, icc1, icc2 = self.ExpectationValue2pt(z, zeta, R, term='icc',
                R_s=R_s, R3=R3, Th=Th, Ts=Ts)
                
            xxcc = iicc - 2. * icc + cc
            
            return xxcc, Rzeros, Rzeros
        
        elif term in ['xxdd', 'nndd']:
            
            if not self.pf['ps_include_4pt']:
                return Rzeros, Rzeros, Rzeros
            
            xx, xx1, xx2 = self.ExpectationValue2pt(z, zeta, R, term='nn',
                R_s=R_s, R3=R3, Th=Th, Ts=Ts)
            dd, dd1, dd2 = self.ExpectationValue2pt(z, zeta, R, term='dd')
            avg_x = self.ExpectationValue1pt(z, zeta, term='n',
                R_s=R_s, Th=Th, Ts=Ts)

            #if self.pf['ps_use_wick']:
            #    xd, xd1, xd2 = self.ExpectationValue2pt(z, zeta, R, term='nd',
            #        R_s=R_s, R3=R3, Th=Th, Ts=Ts)
            #    avg_xd = self.ExpectationValue1pt(z, zeta, term='n*d',
            #        R_s=R_s, Th=Th, Ts=Ts)
            #    xxdd = xx * dd + 2 * avg_xd + xd**2
            #
            #    return xxdd, Rzeros, Rzeros
            #else:
            # xxdd = <xx'dd'> = <dd'(1 - x_i - x_i' + x_i x_i')>
            #      = <dd'> + <x_i x_i' dd'> - 2 <x_i dd'>
            
            # NOTE: if not for the kludge we are applying, we could 
            # simply return xx * dd
            
            idd, idd1, idd2 = self.ExpectationValue2pt(z, zeta, R, term='idd',
                R_s=R_s, R3=R3, Th=Th, Ts=Ts)
            iidd, iidd1, iidd2 = self.ExpectationValue2pt(z, zeta, R, term='iidd',
                R_s=R_s, R3=R3, Th=Th, Ts=Ts)    
                
            xxdd = dd + iidd - 2. * idd
            
            return xxdd, Rzeros, Rzeros

        elif term in ['xxd', 'nnd']:
            # <xx'd'> = <d'> - <x_i d'> - <x_i' d'> + <x_i x_i' d'>
            #         = <x_i x_i' d'> - <x_i d'> - <x_i d>
            
            if not self.pf['ps_include_3pt']:
                return Rzeros, Rzeros, Rzeros
                
            iid, iid1, iid2 = self.ExpectationValue2pt(z, zeta, R, term='iid',
                R_s=R_s, R3=R3, Th=Th, Ts=Ts)
            
            idt, id1, id2 = self.ExpectationValue2pt(z, zeta, R, term='id',
                R_s=R_s, R3=R3, Th=Th, Ts=Ts)
                
            avg_id = self.ExpectationValue1pt(z, zeta, term='i*d',
                R_s=R_s, R3=R3, Th=Th, Ts=Ts)
                
            return iid - idt - avg_id, Rzeros, Rzeros
        
        elif term in ['dd', 'mm']:
            # Equivalent to correlation function since <d> = 0
            return self.spline_cf_mm(z)(np.log(R)), np.zeros_like(R), np.zeros_like(R)
        elif term == 'nd':
            idt, id1, id2 = self.ExpectationValue2pt(z, zeta, R, term='id',
                R_s=R_s, R3=R3, Th=Th, Ts=Ts)
            return -idt, -id1, -id2
        elif term == 'nn':
            # <xx'> = 1 - 2<x_i> + <x_i x_i'>                                                         
            ii, ii1, ii2 = self.ExpectationValue2pt(z, zeta, R, term='ii',
                R_s=R_s, R3=R3, Th=Th, Ts=Ts)
            return 1. - 2. * Qi + ii, Rzeros, Rzeros
        elif term == 'nc':  
            avg_c = self.ExpectationValue1pt(z, zeta, term='c',
                R_s=R_s, Th=Th, Ts=Ts) 
            ic, ic1, ic2 = self.ExpectationValue2pt(z, zeta, R, term='ic',
                R_s=R_s, R3=R3, Th=Th, Ts=Ts)
            return avg_c - ic, avg_c - ic1, avg_c - ic2
        elif term == 'nh':  
            avg_c = self.ExpectationValue1pt(z, zeta, term='h',
                R_s=R_s, Th=Th, Ts=Ts) 
            ih, ih1, ih2 = self.ExpectationValue2pt(z, zeta, R, term='ih',
                R_s=R_s, R3=R3, Th=Th, Ts=Ts)
                
            return avg_c - ih, avg_c - ih1, avg_c - ih2
            
        elif term == 'aa':
            aa = self.CorrelationFunction(z, zeta, R, term='aa',
                R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja, k=k)
                
            return aa, Rzeros, Rzeros
            
        #elif term == 'iid':
        #    idt, id1, id2 = self.ExpectationValue2pt(z, zeta, R, term='id',
        #        R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja, k=k)
        #        
        #    return idt * Qi, Rzeros, Rzeros    
            
        #elif term == 'idd':
        #    return Qi * xi_dd, Rzeros, Rzeros
            

        ##    
        # On to fundamental quantities
        ##
        
        # Some stuff we need
        R_b, M_b, dndm_b = self.BubbleSizeDistribution(z, zeta)
        
        if R_s is None:
            R_s = np.zeros_like(R_b)
        if R3 is None:
            R3 = np.zeros_like(R_b)
        
        Qh = self.BubbleShellFillingFactor(z, zeta, R_s)
        
        delta_b_bar = self.delta_bubble_vol_weighted(z, zeta)
        delta_n_bar = -delta_b_bar * Qi / (1. - Qi)

        # M_bnimum bubble size            
        Mmin = self.Mmin(z) * zeta
        iM = np.argmin(np.abs(M_b - Mmin))
        
        # Only need overlap volumes once per redshift
        all_OV_z = self._cache_Vo(z)
        if all_OV_z is None:
            all_OV_z = np.zeros((len(R), 6, len(R_b)))
            for i, sep in enumerate(R):
                #if i == 10:
                #    print(sep, R_b[0], R_s[0], R_b.size, R_s.size, R3.size)
                all_OV_z[i,:,:] = \
                    np.array(self.overlap_volumes(sep, R_b, R_s, R3))

            self._cache_Vo_[z] = all_OV_z.copy()

            #print("Generated z={} overlap_volumes".format(z))
           
        #else:    
        #   print("Read in z={} overlap_volumes".format(z))
           
        all_IV_z = self._cache_IV(z)
        if all_IV_z is None:
            all_IV_z = np.zeros((len(R), 6, len(R_b)))
            for i, sep in enumerate(R):
                all_IV_z[i,:,:] = \
                    np.array(self.intersectional_volumes(sep, R_b, R_s, R3))
        
            self._cache_IV_[z] = all_IV_z.copy()    
        
        Mmin_b = self.Mmin(z) * zeta
        Mmin_h = self.Mmin(z)
                
        ##
        # Handy
        ##
        if self.pf['ps_include_ion'] and (term != 'ii'):
            _P_ii, _P_ii_1, _P_ii_2 = \
                self.ExpectationValue2pt(z, zeta, R, term='ii', 
                R_s=R_s, R3=R3, Th=Th, Ts=Ts)
        
        #if self.pf['ps_include_temp'] and (term != 'hh'):
        #    _P_hh, _P_hh_1, _P_hh_2 = \
        #        self.ExpectationValue2pt(z, zeta, R, term='hh', 
        #        R_s=R_s, R3=R3, Th=Th, Ts=Ts)
        
        # Loop over scales
        P1 = np.zeros(R.size)
        P2 = np.zeros(R.size)
        PT = np.zeros(R.size)
        for i, sep in enumerate(R):
            
            ##
            # Note: each element of this loop we're constructing an array
            # over bubble mass, which we then integrate over to get a total
            # probability. The shape of every quantity should be `self.m`.
            ##
                        
            # Yields: V11, V12, V13, V22, V23, V33
            # Remember: these radii arrays depend on redshift (through delta_B)
            all_V = all_OV_z[i]
            all_IV = all_IV_z[i]
        
            # For two-halo terms, need bias of sources.
            if self.pf['ps_include_bias']:
                ep = self.excess_probability(z, zeta, sep, term)
            else:
                ep = np.zeros_like(self.m)

            ##
            # For each zone, figure out volume of region where a
            # single source can ionize/heat/couple both points, as well
            # as the region where a single source is not enough (Vss_ne)
            ##
            if term == 'ii':
                
                Vo = all_V[0]
                Vi = 4. * np.pi * R_b**3 / 3.
                            
                # Subtract off more volume if heating is ON.
                #if self.pf['ps_include_temp']:
                #    #Vne1 = Vne2 = Vi - self.IV(sep, R_b, R_s)
                #    Vne1 = Vne2 = Vi - all_IV[1]
                #else:
                
                # You might think: hey! If temperature fluctuations are on,
                # we need to make sure the second point isn't *heated* by
                # the first point. This gets into issues of overlap. By not
                # introducing this correction (commented out above), we're 
                # saying "yes, the second point can still lie in the heated
                # region of the first (ionized) point, but that point itself
                # may actually be ionized, since the way we construct regions
                # doesn't know that a heated region may actually live in the
                # ionized region of another bubble." That is, a heated point
                # can be ionized but an ionized pt can't later be designated
                # a hot point.
                Vne1 = Vne2 = Vi - Vo
                    
                _P1 = self.get_prob(z, M_b, dndm_b, Mmin_b, Vo, True)
                
                _P2_1 = self.get_prob(z, M_b, dndm_b, Mmin_b, Vne1, True)
                _P2_2 = self.get_prob(z, M_b, dndm_b, Mmin_b, Vne2, True, ep)
                
                _P2 = (1. - _P1) * _P2_1 * _P2_2
                                
                P1[i] = _P1
                P2[i] = _P2
                
            elif term == 'hh':
                
                #Vii = all_V[0]
                #_integrand1 = dndm * Vii
                #
                #_exp_int1 = np.exp(-simps(_integrand1[iM:] * M_b[iM:],
                #    x=np.log(M_b[iM:])))
                #_P1_ii = (1. - _exp_int1)
                
                # Region in which two points are heated by the same source
                Vo = all_V[3]
                
                # These are functions of mass
                V1 = 4. * np.pi * R_b**3 / 3.
                V2 = 4. * np.pi * R_s**3 / 3.
                Vh = 4. * np.pi * (R_s**3 - R_b**3) / 3.
                                                                
                # Subtract off region of the intersection HH volume
                # in which source 1 would do *anything* to point 2.
                #Vss_ne_1 = Vh - (Vo - self.IV(sep, R_b, R_s) + all_V[0])
                #Vne1 = Vne2 = Vh - Vo
                # For ionization, this is just Vi - Vo
                Vne1 = V2 - all_IV[2] - (V1 - all_IV[1])
                #Vne1 =  V2 - self.IV(sep, R_s, R_s) - (V1 - self.IV(sep, R_b, R_s))
                Vne2 = Vne1
                
                # Shouldn't max(Vo) = Vh?
                
                #_P1, _P2 = self.get_prob(z, zeta, Vo, Vne1, Vne2, corr, term)
                
                _P1 = self.get_prob(z, M_b, dndm_b, Mmin_b, Vo, True)                
                
                _P2_1 = self.get_prob(z, M_b, dndm_b, Mmin_b, Vne1, True)
                _P2_2 = self.get_prob(z, M_b, dndm_b, Mmin_b, Vne2, True, ep)
                
                _P2 = (1. - _P1) * _P2_1 * _P2_2
                
                # The BSD is normalized so that its integral will recover
                # zeta * fcoll.
                                
                #if i == len(R) - 1:
                #    print('-'*30)
                #    print('hh')
                #    print(Qh**2, _P2, _P1)
                #    print(V1[0], V2[0], Vh[0], Vo[0], all_V[0][0], all_V[3][0])
                #    print(Vne1[0], R_s[0] / R_b[0], V2[0] / V1[0], Vh[0] / V1[0])
                #    print(ep[-1])
                #    #print((V1 - all_V[0]) / V1)
                #    #print(Vh / V1)
                #    #print(Vo / V1)
                #    #print(Vne1 / V1)
                #    print('-'*30)                                           
                #                   
                #    raw_input('<enter>')                    
                                                                                                      
                # Start chugging along on two-bubble term   
                if np.any(Vne1 < 0):
                    N = sum(Vne1 < 0)
                    print('R={}: Vss_ne_1 (hh) < 0 {} / {} times'.format(sep, N, len(R_s)))

                # Must correct for the fact that Qi+Qh<=1
                P1[i] = _P1
                P2[i] = max(_P2, Qh**2)                

            elif term == 'ih':
                
                if not self.pf['ps_include_xcorr_ion_hot']:
                    P1[i] = 0.0
                    P2[i] = Qh * Qi
                    continue
                
                #Vo_sh_r1, Vo_sh_r2, Vo_sh_r3 = \
                #    self.overlap_region_shell(sep, R_b, R_s)
                #Vo = 2. * Vo_sh_r2 - Vo_sh_r3
                Vo = all_V[1]
                
                V1 = 4. * np.pi * R_b**3
                V2 = 4. * np.pi * R_s**3
                Vh = 4. * np.pi * (R_s**3 - R_b**3) / 3.

                # Volume in which I ionize but don't heat (or ionize) the other pt.
                Vne1 = V1 - all_IV[1]
                         
                # Volume in which I heat but don't ionize (or heat) the other pt, 
                # i.e., same as the two-source term for <hh'>
                #Vne2 = Vh - self.IV(sep, R_b, R_s)
                Vne2 =  V2 - all_IV[2] - (V1 - all_IV[1])
                #Vne2 =  V2 - self.IV(sep, R_s, R_s) - (V1 - self.IV(sep, R_b, R_s))
        
                if np.any(Vne2 < 0):
                    N = sum(Vne2 < 0)
                    print('R={}: Vss_ne_2 (ih) < 0 {} / {} times'.format(sep, N, len(R_s)))
            
            
                #_P1, _P2 = self.get_prob(z, zeta, Vo, Vne1, Vne2, corr, term)
                
                _P1 = self.get_prob(z, M_b, dndm_b, Mmin_b, Vo, True)
                
                _P2 = (1. - _P1) \
                    * self.get_prob(z, M_b, dndm_b, Mmin_b, Vne1, True) \
                    * self.get_prob(z, M_b, dndm_b, Mmin_b, Vne2, True, ep)
                
                P1[i] = _P1
                P2[i] = min(_P2, Qh * Qi)
            
            elif term == 'iicc':
                continue
                #P1[i] = _P_ii_1[i] * _P_hh_1[i] * avg_c**2
                #P2[i] = _P_ii_2[i] * _P_hh_2[i] * avg_c**2
                
            elif term == 'icc':
                continue
                #P1[i] = _P_ii_1[i] * _P_hh_1[i] * avg_c**2
                #P2[i] = _P_ii_2[i] * _P_hh_2[i] * avg_c**2    
            
            ## 
            # Density stuff from here down
            ##    
            elif term.count('d') > 0:
                
                if not self.pf['ps_include_xcorr_ion_rho']:
                    # These terms will remain zero
                    if term.count('d') == 1:
                        break
                
                ##
                # First, grab a bunch of stuff we'll need.
                ##
                    
                # Ionization auto-correlations
                #Pii, Pii_1, Pii_2 = \
                #    self.ExpectationValue2pt(z, zeta, R, term='ii',
                #    R_s=R_s, R3=R3, Th=Th, Ts=Ts)
                
                Vo = all_V[0]
                Vi = 4. * np.pi * R_b**3 / 3.
                Vne1 = Vi - Vo
                
                # Mean bubble density
                #B = self._B(z, zeta)
                #rho0 = self.cosm.mean_density0
                #delta = M_b / Vi / rho0 - 1.
                
                M_h = self.halos.tab_M
                iM_h = np.argmin(np.abs(self.Mmin(z) - M_h))
                dndm_h = self.halos.tab_dndm[iz_hmf]
                
                # Bias of bubbles and halos
                ##
                bh = self.halos.Bias(z)
                bb = self.bubble_bias(z, zeta)
                xi_dd_r = self.spline_cf_mm(z)(np.log(sep))
                
                bb_bar = self.mean_bubble_bias(z, zeta)
                ep_bh = bh * bb_bar * xi_dd_r

                # Mean density of halos (mass is arbitrary)
                delta_h = self.mean_halo_overdensity(z)
                
                # Volume of halos (within virial radii)
                Rvir = self.halos.VirialRadius(M_h, z) / 1e3 # Convert to Mpc
                Vvir = 4. * np.pi * Rvir**3 / 3.
                
                #_P1_ii = self.get_prob(z, M_b, dndm_b, Mmin_b, Vo, True)
                #delta_b_bar = self.mean_bubble_overdensity(z, zeta)
                
                if term == 'id':
                    ##
                    # Analog of one source or one bubble term is P_in, i.e.,
                    # probability that points are in the same bubble.
                    # The "two bubble term" is instead P_out, i.e., the 
                    # probability that points are *not* in the same bubble.
                    # In the latter case, the density can be anything, while
                    # in the former it will be the mean bubble density.
                    ##
                                        
                    # 
                    #_P1 = self.get_prob(z, M_b, dndm_b, Mmin_b, Vo, True) \
                    #    * delta_b_bar
                    #
                    #_P2_1 = (1. - _P_ii_1[i]) \
                    #    * self.get_prob(z, M_b, dndm_b, Mmin_b, Vne1, True)
                    ##_P2_2 = delta_h \
                    ##    * self.get_prob(z, M_h, dndm_h, 0.0, Vvir, False, ep_bh)
                    #_P2_2 = delta_b_bar \
                    #    * self.get_prob(z, M_b, dndm_b, Mmin_b, Vne1, True, ep_bb)
                    #
                    _P1 = _P_ii_1[i] * delta_b_bar
                    #P2[i] = _P2_1 * (_P2_2 - self.fcoll_vol(z) * delta_h)
                    _P2 = _P_ii_2[i] * delta_b_bar #- (1. - _P_ii_2[i] - _P_ii_1[i]) * delta_n_bar
                    #_P2 = (1. - _P_ii_1[i]) * delta_b_bar \
                    #    * self.get_prob(z, M_b, dndm_b, Mmin_b, Vne1, True) \
                    #    * self.get_prob(z, M_b, dndm_b, Mmin_b, Vne1, True, ep)
                    
                    P1[i] = _P1
                    P2[i] = _P2 + delta_n_bar * (1. - Qi) * Qi
                    
                    #print(sep, np.abs(delta_n_bar * (1. - Qi) * Qi) / np.abs(_P2))
                    
                    # Add in "minihalos" term?
                    
                elif term == 'idd':
                    # Second point can be ionized or neutral.
                                        
                    #_P1 = _P_ii_1[i] * delta_b_bar
                    #B = self._B(z, zeta)
                    _P1 = _P_ii_1[i] * delta_b_bar**2
                    
                    # Probability that only the first point is ionized
                    # ...then more stuff
                    #_P2_1 = (1. - _P_ii_1[i]) \
                    #    * self.get_prob(z, M_b, dndm_b, Mmin_b, Vne1, True) \
                    #    * delta_b_bar
                    #_P2_2 = delta_b_bar \
                    #    * self.get_prob(z, M_h, dndm_h, 0.0, Vvir, False, ep)
                    
                    _P2 = (1. - _P_ii_1[i]) \
                        * delta_b_bar**2 \
                        * self.get_prob(z, M_b, dndm_b, Mmin_b, Vne1, True) \
                        * self.get_prob(z, M_b, dndm_b, Mmin_b, Vne1, True, ep)
                    
                    
                    P1[i] = _P1
                    P2[i] = _P2 \
                          + delta_b_bar * delta_n_bar * (1. - Qi) * Qi
                    
                    # + delta_n_bar * (1. - Qi) * Qi?
                    
                    #P2[i] = - 2. * Qi * xi_dd
                    
                    #P1[i] = _P_ii_1[i] * delta_b_bar**2
                    ##P2[i] = _P2_1 * (_P2_2 - self.fcoll_vol(z) * delta_h)
                    #P2[i] = _P_ii_2[i] * delta_b_bar**2    
                
                elif term == 'iid':
                    # This is like the 'id' term except the second point
                    # has to be ionized. 
  
                    _P1 = _P_ii_1[i] * delta_b_bar
                    #B = self._B(z, zeta)
                    #_P1 = delta_b_bar \
                     #   * self.get_prob(z, M_b, dndm_b, Mmin_b, Vo, True) \
                        

                    _P2 = (1. - _P_ii_1[i]) * delta_b_bar \
                        * self.get_prob(z, M_b, dndm_b, Mmin_b, Vne1, True) \
                        * self.get_prob(z, M_b, dndm_b, Mmin_b, Vne1, True, ep)
                        #* self.get_prob(z, M_h, dndm_h, Mmin_h, Vvir, False, ep_bb)

                    P1[i] = _P1
                    P2[i] = _P2

                                    
                elif term == 'iidd':
                    #continue
                    P1[i] = _P_ii_1[i] * xi_dd[i]
                    P2[i] = _P_ii_2[i] * xi_dd[i]
                    
                  

            #elif term == 'hd':
            #    Vo = all_V[3]
            #    
            #    delta_B = self._B(z, zeta, zeta)
            #    _Pin_int = dndm * Vo * M_b * delta_B
            #    Pin = np.trapz(_Pin_int[iM:], x=np.log(M_b[iM:]))
            #    
            #    Vne1 = Vne2 = np.zeros_like(R_b)
            #                
            #    #xi_dd = data['xi_dd_c'][i]
            #    #
            #    #bHII = self.bubble_bias(z, zeta)
            #                    
            #    bh = self.halos.Bias(z)
            #    bb = self.bubble_bias(z, zeta)
            #    
            #    Vi = 4. * np.pi * (R_s**3 - R_b**3) / 3.
            #    _prob_i_d = dndm * bb * xi_dd[i] * Vi * M_b
            #    prob_i_d = np.trapz(_prob_i_d[iM:], x=np.log(M_b[iM:]))
            #    
            #    Pout = prob_i_d #+ data['Qh']
            #    
            #    print(z, i, sep, Pin, Pout)
            #    
            #    limiter = None
                
            else:
                raise NotImplementedError('No method in place to compute <{}\'>'.format(term))
        
        ##
        # Kludges!
        # Note: won't make it here if not ionization field is random.
        ##
        
        # To ensure vanishing fluctuations as R->inf, must
        # augment <x_i x_i' d d'> term.
        kludge = 0.0
        
        #if self.pf['ps_include_xcorr_ion_rho']:
            
        if term == 'iidd':
            
            
            if self.pf['ps_include_ion']:
                avg_id = self.ExpectationValue1pt(z, zeta, term='i*d')
                                        
                kludge += avg_id**2
                if self.pf['ps_use_wick']:
                    idt, id1, id2 = \
                        self.ExpectationValue2pt(z, zeta, R, term='id')
                    kludge += idt**2
                    
        if term == 'iicc':
            
            
            if self.pf['ps_include_ion']:
                avg_c = self.ExpectationValue1pt(z, zeta, term='c')
                avg_xd = self.ExpectationValue1pt(z, zeta, term='n*d')
            
                kludge += 2 * avg_xd * avg_c
               #if self.pf['ps_use_wick']:
               #    idt, id1, id2 = \
               #        self.ExpectationValue2pt(z, zeta, R, term='id')
               #    kludge += idt**2                
               #
        #if self.pf['ps_include_xcorr_ion_hot']:
        #elif term == 'iicc':
        #    print('hello')
        #    avg_xc = self.ExpectationValue1pt(z, zeta, term='n*c',
        #        R_s=R_s, R3=R3, Th=Th, Ts=Ts)
        #    avg_c = self.ExpectationValue1pt(z, zeta, term='c',
        #        R_s=R_s, R3=R3, Th=Th, Ts=Ts)
        #    kludge += avg_c**2 #- Qi**2 * avg_c**2 #- 2 * Qi**2 * avg_c
        
        #elif term == 'iicc':
        #    #avg_xc = self.ExpectationValue1pt(z, zeta, term='n*c',
        #    #    R_s=R_s, R3=R3, Th=Th, Ts=Ts)
        #    avg_c = self.ExpectationValue1pt(z, zeta, term='c',
        #        R_s=R_s, R3=R3, Th=Th, Ts=Ts)
        #    kludge += avg_c**2 - Qi**2 * avg_c**2 - 2 * Qi**2 * avg_c
              
                    
        PT = P1 + P2
        
        #if (term == 'hh') and (Qh > 0.5):
        #    PT -= P2
                    
        PT += kludge
                        
        
        ##
        # This needs more thought.
        ##
        #if self.pf['ps_volfix']:
        #    if (term == 'ii') and (Qi >= 0.5):
        #        PT = (1. - Qi) * P1 + Qi**2
        #    elif term in ['id']:
        #        if (Qi >= 0.5):
        #            PT = P1 + P2
        #        else:
        #            PT = (1. - Qi) * P1
        #    elif term in ['idd']:
        #        if (Qi >= 0.5):
        #            PT = P1 + P2
        #        else:
        #            PT = (1. - Qi) * P1
        #    elif term == 'iidd':
        #        
        #        kludge = 0.0
        #        if self.pf['ps_include_ion']:
        #            avg_id = self.ExpectationValue1pt(z, zeta, term='i*d')
        #            kludge += avg_id**2
        #            if self.pf['ps_use_wick']:
        #                idt, id1, id2 = \
        #                    self.ExpectationValue2pt(z, zeta, R, term='id')
        #                kludge += avg_id**2 + idt**2
        #            
        #        print(kludge)
        #            
        #        if Qi < 0.5:
        #            PT = _P_ii * xi_dd + kludge
        #        else:
        #            PT = (1. - Qi) * P1
            

        self._cache_jp_[z][term] = R, PT, P1, P2
        
        return PT, P1, P2
        
    def get_prob(self, z, M, dndm, Mmin, V, exp=True, ep=0.0, Mmax=None):
        """
        Basically do an integral over some distribution function.
        """
        
        # Set lower integration limit
        iM = np.argmin(np.abs(M - Mmin))
        
        if Mmax is not None:
            iM2 = np.argmin(np.abs(M - Mmax)) + 1
        else:
            iM2 = None    
        
        # One-source term
        integrand = dndm * V * (1. + ep)
                 
        integr = simps(integrand[iM:iM2] * M[iM:iM2], x=np.log(M[iM:iM2])) 
        
        # Exponentiate?
        if exp:
            exp_int = np.exp(-integr)
            P = 1. - exp_int
        else:
            P = integr
        
        return P
        
    def CorrelationFunction(self, z, zeta=None, R=None, term='ii', 
        R_s=None, R3=0.0, Th=500., Tc=1., Ts=None, k=None, Tk=None, Ja=None):
        """
        Compute the correlation function of some general term.
        
        """
        
        Qi = self.MeanIonizedFraction(z, zeta)
        Qh = self.BubbleShellFillingFactor(z, zeta, R_s)
        
        if R is None:
            use_R_tab = True
            R = self.halos.tab_R
        else:
            use_R_tab = False    
            
        Tcmb = self.cosm.TCMB(z)
        Tgas = self.cosm.Tgas(z)           
       
        ##
        # Check cache for match
        ##
        cached_result = self._cache_cf(z, term)
        
        if cached_result is not None:
            _R, _cf = cached_result
            
            if _R.size == R.size:
                if np.allclose(_R, R):
                    return _cf
            
            return np.interp(R, _R, _cf)
 
        ##
        # 21-cm correlation function
        ##  
        if term in ['21', 'phi', 'psi']:
            
            #if term in ['21', 'phi']:
            #    ev_2pt, ev_2pt_1, ev_2pt_2 = \
            #        self.ExpectationValue2pt(z, zeta, R=R, term='phi',
            #            R_s=R_s, R3=R3, Th=Th, Ts=Ts)
            #    avg_phi = self.ExpectationValue1pt(z, zeta, term='phi',
            #        R_s=R_s, Th=Th, Ts=Ts)
            #
            #    cf_21 = ev_2pt - avg_phi**2
            #
            #else:
            ev_2pt, ev_2pt_1, ev_2pt_2 = \
                self.ExpectationValue2pt(z, zeta, R=R, term='psi',
                    R_s=R_s, R3=R3, Th=Th, Ts=Ts)
            avg_psi = self.ExpectationValue1pt(z, zeta, term='psi',
                R_s=R_s, Th=Th, Ts=Ts)
            
            cf_psi = ev_2pt - avg_psi**2
            
            ##
            # Temperature fluctuations
            ##
            include_temp = self.pf['ps_include_temp']
            include_lya =  self.pf['ps_include_lya']
            if (include_temp or include_lya) and term in ['phi', '21']:
                
                Phi, Phi1, Phi2 = self.ExpectationValue2pt(z, zeta, R=R, 
                    term='Phi', R_s=R_s, Ts=Ts, Tk=Tk, Th=Th, Ja=Ja, k=k)
                avg_phi = self.ExpectationValue1pt(z, zeta, term='phi', 
                    R_s=R_s, Ts=Ts, Tk=Tk, Th=Th, Ja=Ja, R3=R3)
                    
                cf_21 = cf_psi + Phi - (avg_phi**2 - avg_psi**2)
                                
            else:
                cf_21 = cf_psi

            cf = cf_21
            
        elif term == 'nn':                                                         
            cf = -self.CorrelationFunction(z, zeta, R, term='ii',
                R_s=R_s, R3=R3, Th=Th, Ts=Ts)
            return cf
        elif term == 'nc':                                                         
            cf = -self.CorrelationFunction(z, zeta, R, term='ic',
                R_s=R_s, R3=R3, Th=Th, Ts=Ts)
            return cf  
        elif term == 'nd':                                                         
            cf = -self.CorrelationFunction(z, zeta, R, term='id',
                R_s=R_s, R3=R3, Th=Th, Ts=Ts)
            return cf
              
        
        ##
        # Matter correlation function -- we have this tabulated already.
        ##    
        elif term in ['dd', 'mm']:
            
            if not self.pf['ps_include_density']:
                cf = np.zeros_like(R)    
                self._cache_cf_[z][term] = R, cf
                return cf
            
            iz = np.argmin(np.abs(z - self.halos.tab_z_ps))
            if use_R_tab:
                cf = self.halos.tab_cf_mm[iz]
            else:
                cf = np.interp(np.log(R), np.log(self.halos.tab_R), 
                    self.halos.tab_cf_mm[iz])

        ##
        # Ionization correlation function
        ##
        elif term == 'ii':
            if not self.pf['ps_include_ion']:
                cf = np.zeros_like(R)    
                self._cache_cf_[z][term] = R, cf
                return cf
                
            ev_ii, ev_ii_1, ev_ii_2 = \
                self.ExpectationValue2pt(z, zeta, R=R, term='ii',
                    R_s=R_s, R3=R3, Th=Th, Ts=Ts)
    
            ev_i = self.ExpectationValue1pt(z, zeta, term='i',
                R_s=R_s, Th=Th, Ts=Ts)
            cf = ev_ii - ev_i**2
            
        ##
        # Temperature correlation function
        ##
        elif term == 'hh':
            if not self.pf['ps_include_temp']:
                cf = np.zeros_like(R)    
                self._cache_cf_[z][term] = R, cf
                return cf
        
            jp_hh, jp_hh_1, jp_hh_2 = \
                self.ExpectationValue2pt(z, zeta, R=R, term='hh', 
                    R_s=R_s, R3=R3, Th=Th, Ts=Ts)
        
            #if self.pf['ps_volfix']:
            #    if (Qh < 0.5):
            #        ev_hh = jp_hh_1 + jp_hh_2
            #    else:
            #        # Should this 1-Qh factor be 1-Qh-Qi?
            #        ev_hh = (1. - Qh) * jp_hh_1 + Qh**2
            #else:    
            #    ev_hh = jp_hh 
        
            # Should just be Qh
            Qh = self.BubbleShellFillingFactor(z, zeta, R_s)
            ev_h = self.ExpectationValue1pt(z, zeta, term='h',
                R_s=R_s, Ts=Ts, Th=Th)
        
            cf = jp_hh - ev_h**2    
            
        ##
        # Ionization-density cross correlation function
        ##
        elif term == 'id':
            if not self.pf['ps_include_xcorr_ion_rho']:
                cf = np.zeros_like(R)    
                self._cache_cf_[z][term] = R, cf
                return cf

            #jp_ii, jp_ii_1, jp_ii_2 = \
            #    self.ExpectationValue2pt(z, zeta, R, term='ii',
            #    R_s=R_s, R3=R3, Th=Th, Ts=Ts)

            jp_im, jp_im_1, jp_im_2 = \
                self.ExpectationValue2pt(z, zeta, R, term='id',
                R_s=R_s, R3=R3, Th=Th, Ts=Ts)
                
            ev_xd = 0.0

            # Add optional correction to ensure limiting behavior?        
            if self.pf['ps_volfix']:
                #if Qi < 0.5:
                ev = jp_im_1 + jp_im_2
                #else:
                #    ev = jp_im_1 - jp_ii_1
            else:    
                ev = jp_im

            # Equivalent to correlation function in this case.
            cf = ev - ev_xd
                        
        # c = contrast, instead of 'c' for cold, use '3' for zone 3 (later)    
        elif term == 'cc':
            if not self.pf['ps_include_temp']:
                cf = np.zeros_like(R)    
                self._cache_cf_[z][term] = R, cf
                return cf
                
            ev_cc, ev_cc_1, ev_cc_2 = \
                self.ExpectationValue2pt(z, zeta, R=R, term='cc', 
                    R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)
                        
            ev_c = self.ExpectationValue1pt(z, zeta, term='c',
                R_s=R_s, Ts=Ts, Th=Th, Tk=Tk, Ja=Ja)
            
            #else:
            #    # Remember, this is for the hot/cold term
            #    Csq = (Tcmb / (Tk - Tcmb)) * delta_T[0] * delta_T[1] \
            #        / (1. + delta_T[0]) / (1. + delta_T[1])
            #    C = np.sqrt(C)    
            
                
            # Add optional correction to ensure limiting behavior?        
            #if self.pf['ps_volfix']:
            #    if (Qh < 0.5) and (Qi < 0.5):
            #        ev_cc = jp_cc_1 + jp_cc_2
            #    else:
            #        ev_cc = (1. - Qi) * jp_cc_1 + ev_c**2
            #else:    
            #    ev_cc = jp_cc
                           
            cf = ev_cc - ev_c**2
            
        ##
        # Ionization-heating cross-correlation function
        ##    
        elif term == 'ih':
            if not self.pf['ps_include_temp']:
                cf = np.zeros_like(R)    
                self._cache_cf_[z][term] = R, cf
                return cf
                
            ev_2pt, ev_1, ev_2 = \
                self.ExpectationValue2pt(z, zeta, R=R, term='ih',
                    R_s=R_s, R3=R3, Th=Th, Ts=Ts)
                
            # Add optional correction to ensure limiting behavior?        
            #if self.pf['ps_volfix']:
            #    if (Qh < 0.5) and (Qi < 0.5):
            #        ev_2pt = jp_1 + jp_2
            #    else:
            #        ev_2pt = (1. - Qh) * jp_1 + Qh * Qi
            #else:    
            #    ev_2pt = jp + Qh * Qi
        
            ev_1pt_i = self.ExpectationValue1pt(z, zeta, term='i', 
                R_s=R_s, Ts=Ts, Th=Th)
            ev_1pt_h = self.ExpectationValue1pt(z, zeta, term='h', 
                R_s=R_s, Ts=Ts, Th=Th)    

            cf = ev_2pt - ev_1pt_i * ev_1pt_h

        elif term == 'ic':
            ev_2pt, ev_1, ev_2 = \
                self.ExpectationValue2pt(z, zeta, R=R, term='ic',
                    R_s=R_s, R3=R3, Th=Th, Ts=Ts)

            # Add optional correction to ensure limiting behavior?        
            #if self.pf['ps_volfix']:
            #    if (Qh < 0.5) and (Qi < 0.5):
            #        ev_2pt = jp_1 + jp_2
            #    else:
            #        ev_2pt = (1. - Qh) * jp_1 + Qh * Qi
            #else:    
            #    ev_2pt = jp + Qh * Qi
        
            ev_1pt = self.ExpectationValue1pt(z, zeta, term='i*c', 
                R_s=R_s, Ts=Ts, Th=Th)

            cf = ev_2pt - ev_1pt    
        
        ##
        # Special case: Ly-a
        ##
        elif term == 'aa':
            Mmin = lambda zz: self.Mmin(zz)

            # Horizon set by distance photon can travel between n=3 and n=2
            zmax = self.hydr.zmax(z, 3)
            rmax = self.cosm.ComovingRadialDistance(z, zmax) / cm_per_mpc
        
            # Light-cone effects?
            if self.pf['ps_include_lya_lc']:
                
                # Use specific mass accretion rate of Mmin halo
                # to get characteristic halo growth time. This is basically
                # independent of mass so it should be OK to just pick Mmin.
                
                if type(self.pf['ps_include_lya_lc']) is float:
                    a = lambda zz: self.pf['ps_include_lya_lc']
                else:
                
                    #oot = lambda zz: self.pops[0].dfcolldt(z) / self.pops[0].halos.fcoll_2d(zz, np.log10(Mmin(zz)))
                    #a = lambda zz: (1. / oot(zz)) / pop.cosm.HubbleTime(zz)                        
                    oot = lambda zz: self.halos.MAR_func(zz, Mmin(zz)) / Mmin(zz) / s_per_yr
                    a = lambda zz: (1. / oot(zz)) / self.cosm.HubbleTime(zz)
                
                tstar = lambda zz: a(zz) * self.cosm.HubbleTime(zz)
                rstar = c * tstar(z) * (1. + z) / cm_per_mpc
                uisl = lambda kk, mm, zz: self.halos.u_isl_exp(kk, mm, zz, rmax, rstar)
            else:
                uisl = lambda kk, mm, zz: self.halos.u_isl(kk, mm, zz, rmax)
                            
            ps_try = self._cache_ps(z, 'aa')
            
            if ps_try is not None:
                ps = ps_try
            else:
                ps = np.array([self.halos.PowerSpectrum(z, _k, uisl, Mmin(z)) \
                    for _k in k])
                self._cache_ps_[z][term] = ps
                
            cf = self.CorrelationFunctionFromPS(R, ps, k, split_by_scale=True)
                    
        else:
            raise NotImplementedError('Unrecognized correlation function: {}'.format(term))
        
        #if term not in ['21', 'mm']:
        #    cf /= (2. * np.pi)**3
        
        self._cache_cf_[z][term] = R, cf
        
        return cf
            
   # def PowerSpectrum(self, z, zeta, Q=None, term='ii', rescale=False, 
   #     cf=None, R=None):
   #     
   #     if cf is None:
   #         cf = self.CorrelationFunction(z, zeta, Q=Q, term=term, 
   #             rescale=rescale)
   #     else:
   #         
   #         if R is None:
   #             R = self.halos.tab_R
   #         
   #         assert cf.size == R.size
   #         print('Correlation function supplied. Neglecting all other parameters.')    
   #         
   #     # Integrate over R
   #     func = lambda k: self.halos._integrand_FT_3d_to_1d(cf, k, R)
   #       
   #     return np.array([np.trapz(func(k) * R, x=np.log(R)) \
   #         for k in self.halos.tab_k]) / 2. / np.pi
   
    def TempToContrast(self, z, Th=500., Ts=None):
        
        if Th is None:
            return 0.0
        
        Tcmb = self.cosm.TCMB(z)
        Tgas = self.cosm.Tgas(z)
        
        if Ts is None:
            print("Assuming Ts=Tgas(unheated).")
            Ts = Tgas
        
        delta_T = Ts / Th - 1.
            
        #if ii <= 1:

        # Contrast of hot regions.
        return (Tcmb / (Th - Tcmb)) * delta_T / (1. + delta_T)

    def CorrelationFunctionFromPS(self, R, ps, k=None, split_by_scale=False,
        kmin=None, epsrel=1-8, epsabs=1e-8, method='clenshaw-curtis', 
        use_pb=False, suppression=np.inf):
        return self.halos.InverseFT3D(R, ps, k, kmin=kmin, 
            epsrel=epsrel, epsabs=epsabs, use_pb=use_pb,
            split_by_scale=split_by_scale, method=method, suppression=suppression)
            
    def PowerSpectrumFromCF(self, k, cf, R=None, split_by_scale=False,
        Rmin=None, epsrel=1-8, epsabs=1e-8, method='clenshaw-curtis',
        use_pb=False, suppression=np.inf):
        return self.halos.FT3D(k, cf, R, Rmin=Rmin, 
            epsrel=epsrel, epsabs=epsabs, use_pb=use_pb,
            split_by_scale=split_by_scale, method=method, suppression=suppression)
        
