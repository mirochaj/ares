"""

FluctuatingBackground.py

Author: Jordan Mirocha
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

    def _overlap_region(self, dr, R1, R2):
        """
        Volume of intersection between two spheres of radii R1 < R2.
        """
        if dr >= (R1 + R2):
            return 0.0         
        elif dr <= (R2 - R1):
            # This means the points are so close that the overlap region
            # of the outer spheres completely engulfs the inner sphere.
            # Not obvious why the general formula doesn't recover this limit.
            return 4. * np.pi * R1**3 / 3.
        else:
            return np.pi * (R2 + R1 - dr)**2 \
                * (dr**2 + 2. * dr * R1 - 3. * R1**2 \
                 + 2. * dr * R2 + 6. * R1 * R2 - 3. * R2**2) / 12. / dr
    
    def IV(self, *args):
        """
        Just a vectorized version of the overlap calculation.
        """
        if not hasattr(self, '_IV'):
            self._IV = np.vectorize(self._overlap_region)
        return self._IV(*args)
    
    def intersectional_volume(self, dr, R1, R2):
        return self.IV(dr, R1, R2)
    
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
    
    def BubbleShellFillingFactor(self, z, zeta, Rh):
        """
        
        """
        if self.pf['bubble_size_dist'] is None:
            R_b = self.pf['bubble_size']
            V_b = 4. * np.pi * R_b**3 / 3.
            n_b = self.BubbleDensity(z)
        
            Qh = 1. - np.exp(-n_b * V_b)
        elif self.pf['bubble_size_dist'].lower() == 'fzh04':
            Ri, Mi, dndm = self.BubbleSizeDistribution(z, zeta)
            Q = self.MeanIonizedFraction(z, zeta)
            
            if type(Rh) is np.ndarray:
                nz = Ri > 0
                assert np.allclose(np.diff(Rh[nz==1] / Ri[nz==1]), 0.0), \
                    "No support for absolute scaling of hot bubbles yet."
                    
                Qh = Q * ((Rh[0] - Ri[0]) / Ri[0])**3
                    
                return np.minimum(Qh, 1. - Q)
            else:
                Qh = 0.
            
            #if np.logical_and(np.all(Rh == 0), np.all(Rc == 0)):
            #    return 0.
            #
            #Mmin = self.Mmin(z)
            #iM = np.argmin(np.abs(Mmin * zeta - Mi))
            #
            #Vi = 4. * np.pi * Ri**3 / 3.   
            #Vsh1 = 4. * np.pi * (Rh - Ri)**3 / 3.
            #
            #dndlnm = dndm * Mi
            #
            #if np.all(Rc == 0):
            #    Vsh2 = Qc = 0.0
            #else:
            #    Vsh2 = 4. * np.pi * (Rc - Rh)**3 / 3.
            #    Qc = np.trapz(dndlnm[iM:] * Vsh2[iM:], x=np.log(Mi[iM:]))
            #
            #Qi = np.trapz(dndlnm[iM:] * Vi[iM:], x=np.log(Mi[iM:]))
            #Qh = np.trapz(dndlnm[iM:] * Vsh1[iM:], x=np.log(Mi[iM:]))
            
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
        
        return zeta * self.halos.fcoll_2d(z, logM)

    def BubbleFillingFactor(self, z, zeta, rescale=True):
        """
        Fraction of volume filled by bubbles.
        """
        

        if self.bsd_model is None:
            Ri = self.pf['bubble_size']
            Vi = 4. * np.pi * R_b**3 / 3.
            ni = self.BubbleDensity(z)

            Qi = 1. - np.exp(-ni * Vi)

        elif self.bsd_model in ['fzh04', 'hmf']:
            
            # Smallest bubble is one around smallest halo.
            # Don't actually need its mass, just need index to correctly
            # truncate integral.
            Mmin = self.Mmin(z)
            logM = np.log10(Mmin)

            # Mi should just be self.m? No.
            Ri, Mi, dndm = self.BubbleSizeDistribution(z, zeta, rescale)
            Vi = 4. * np.pi * Ri**3 / 3.
            
            dndlnm = dndm * Mi
            
            iM = np.argmin(np.abs(Mmin * zeta - Mi))
            
            Qi = simps(dndlnm[iM:] * Vi[iM:], x=np.log(Mi[iM:]))
            
            # This means reionization is over.
            if self.bsd_model == 'fzh04':
                if self._B0(z, zeta) <= 0:
                    return 1.
                
        else:
            raise NotImplemented('Uncrecognized option for BSD.')
        
        return min(Qi, 1.)
        
        # Grab heated phase to enforce BC
        #Rs = self.BubbleShellRadius(z, Ri)        
        #Vsh = 4. * np.pi * (Rs - Ri)**3 / 3.
        #Qh = np.trapz(dndm * Vsh * Mi, x=np.log(Mi))   
        
        #if lya and self.pf['bubble_pod_size_func'] in [None, 'const', 'linear']:
        #    Rc = self.BubblePodRadius(z, Ri, zeta, zeta_lya)
        #    Vc = 4. * np.pi * (Rc - Ri)**3 / 3.
        #    
        #    if self.pf['powspec_rescale_Qlya']:
        #        # This isn't actually correct since we care about fluxes
        #        # not number of photons, but fine for now.
        #        Qc = min(zeta_lya * self.halos.fcoll_2d(z, np.log10(self.Mmin(z))), 1)
        #    else:
        #        Qc = np.trapz(dndlnm[iM:] * Vc[iM:], x=np.log(Mi[iM:]))
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

    def bubble_bias(self, z, zeta):
        """
        Eq. 9.24 in Loeb & Furlanetto (2013).
        """
        iz = np.argmin(np.abs(z - self.halos.tab_z))
        s = self.sigma
        S = s**2

        return 1. + ((self._B(z, zeta, zeta)**2 / S - (1. / self._B0(z, zeta))) \
            / self._growth_factor(z))

    def mean_bubble_bias(self, z, zeta, term='ii'):
        """
        Note that we haven't yet divided by QHII!
        """
        
        Ri, Mi, dndm = self.BubbleSizeDistribution(z, zeta)
    
        #if ('h' in term) or ('c' in term) and self.pf['powspec_temp_method'] == 'shell':
        #    Rh, Rc = self.BubbleShellRadius(z, Ri)
        #    R = Rh
        #else:
        R = Ri
    
        V = 4. * np.pi * R**3 / 3.
    
        Mmin = self.Mmin(z) * zeta
        iM = np.argmin(np.abs(Mmin - self.m))    
        bHII = self.bubble_bias(z, zeta)
        
        #tmp = dndm[iM:]
        #print(z, len(tmp[np.isnan(tmp)]), len(bHII[np.isnan(bHII)]))
            
        #imax = int(min(np.argwhere(np.isnan(Ri))))
    
        return simps(dndm[iM:] * V[iM:] * bHII[iM:] * Mi[iM:],
            x=np.log(Mi[iM:]))
    
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
        
        # For 'hh' term, should we modify R? Probably a small effect...
        Q = self.BubbleFillingFactor(z, zeta)
        
        # Function of bubble mass (bubble size)
        bHII = self.bubble_bias(z, zeta)
        bbar = self.mean_bubble_bias(z, zeta, term) / Q
        
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
        return np.interp(z, self.halos.tab_z, self.halos.tab_growth)
    
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

    def BubblePodSizeDistribution(self, z, zeta):
        if self.pf['powspec_lya_method'] == 1:
            # Need to modify zeta and critical threshold
            Rc, Mc, dndm = self.BubbleSizeDistribution(z, zeta)
            return Rc, Mc, dndm
        else:
            raise NotImplemented('help please')

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
                Ri = self.pf['bubble_size']
                Mi = (4. * np.pi * Rb**3 / 3.) * rho0_m
                dndm = self.pf['bubble_density']
            else:
                raise NotImplementedError('help')
        
        elif self.bsd_model == 'hmf':
            Mi = self.halos.tab_M * zeta
            # Assumes bubble material is at cosmic mean density
            Ri = (3. * Mi / rho0_b / 4. / np.pi)**(1./3.)
            iz = np.argmin(np.abs(z - self.halos.tab_z))
            dndm = self.halos.tab_dndm[iz].copy()
        
        elif self.bsd_model == 'fzh04':
            # Just use array of halo mass as array of ionized region masses.
            # Arbitrary at this point, just need an array of masses.
            # Plus, this way, the sigma's from the HMF are OK.
            Mi = self.m 
                                    
            # Radius of ionized regions as function of delta (mass)
            Ri = (3. * Mi / rho0_m / (1. + delta_B) / 4. / np.pi)**(1./3.)
        
            Vi = four_pi * Ri**3 / 3.
            
            # This is Eq. 9.38 from Steve's book.
            # The factors of 2, S, and Mi are from using dlns instead of 
            # dS (where S=s^2)
            dndm = rho0_m * self.pcross(z, zeta) * 2 * np.abs(self.dlns_dlnm) \
                * self.sigma**2 / Mi**2
                
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
            iM = np.argmin(np.abs(Mi - Mmin))
            Q = simps(dndm[iM:] * Vi[iM:] * Mi[iM:], x=np.log(Mi[iM:]))
            xibar = self.MeanIonizedFraction(z, zeta)
            dndm *= -np.log(1. - xibar) / Q
            
        return Ri, Mi, dndm
        
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

    def get_Nion(self, z, Ri):
        return 4. * np.pi * (Ri * cm_per_mpc / (1. + z))**3 \
            * self.cosm.nH(z) / 3.
            
    def BubblePodRadius(self, z, Ri, zeta=40., zeta_lya=1.):
        """
        Calling this a 'pod' since it may one day contain multiple bubbles.
        """
        
        #if not self.pf['include_lya_fl']:
        #    return np.zeros_like(Ri)
        
        # This is basically assuming self-similarity
        if self.pf['bubble_pod_size_func'] is None:
            if self.pf['bubble_pod_size_rel'] is not None:
                return Ri * self.pf['bubble_pod_size_rel']
            elif self.pf['bubble_pod_size_abs'] is not None:
                return Ri + self.pf['bubble_pod_size_abs']
            else:
                Mi = (4. * np.pi * Ri**3 / 3.) * self.cosm.mean_density0
                Mc = Mi * (zeta_lya / zeta) * self.pf['bubble_pod_Nsc']
                Rc = ((Mc / self.cosm.mean_density0) * 0.75 / np.pi)**(1./3.)

        # 
        elif self.pf['bubble_pod_size_func'] == 'const':
            # Number of ionizing photons in this region, i.e.,
            # integral of photon production over time.
            Nlyc = 4. * np.pi * (Ri * cm_per_mpc / (1. + z))**3 * self.cosm.nH(z) / 3.
            Nlya = (zeta_lya / zeta) * Nlyc
            
            tmax = self.cosm.LookbackTime(z, 50.)
            Ndot = Nlya / tmax

            # This is the radius at which the number density of Ly-a photons is
            # equal to the number density of hydrogen atoms.
            #r_p = np.sqrt(Ndot / c / self.cosm.nH(z) / 4. / np.pi) / cm_per_mpc
            #r_c = r_p * (1. + z)
            #
            #Rc = r_c#max(Ri, r_c)
            
            # Cast in terms of flux
            Ndot /= dnu
            Sa = 1.
            Jc = 5.5e-12 * (1. + z) / Sa
            #Tk = 10.
            #lw = np.sqrt(8. * k_B * Tk * np.log(2.) / m_p / c**2)
            r_p = np.sqrt(Ndot / 4. / np.pi / Jc) / cm_per_mpc
            Rc = r_p * (1. + z)
            
            
        elif self.pf['bubble_pod_size_func'] == 'linear':
            
            Nlyc = 4. * np.pi * (Ri * cm_per_mpc / (1. + z))**3 * self.cosm.nH(z) / 3.
            Nlya = (zeta_lya / zeta) * Nlyc

            t0 = self.cosm.t_of_z(50.)
            t = self.cosm.t_of_z(z)
            
            dt = t - t0
            r_lt = c * dt / cm_per_mpc
            
            Rc = []
            for k, _Nlya in enumerate(Nlya):
            
                norm = _Nlya / (((t - t0)**2 - t0**2) / 2.)
                
                Ndot = lambda t: norm * (t - t0)
                
                # This is the radius at which the number of Ly-a photons i
                # equal to the number of hydrogen atoms.
                Sa = 1.
                Jc = 5.5e-12 * (1. + z) / Sa
            
                # 
                to_solve = lambda r: np.abs(Ndot(t - r * cm_per_mpc / c) / dnu \
                    / 4. / np.pi / (r * cm_per_mpc)**2 - Jc)
                r_p = fsolve(to_solve, Ri[k]*(1.+z), factor=0.1, maxfev=10000)[0]
                r_c = np.minimum(r_p, r_lt) * (1. + z)
            
                Rc.append(r_c)
            
            Rc = np.array(Rc)
                        
        elif self.pf['bubble_pod_size_func'] == 'exp':

            Nlyc = 4. * np.pi * (Ri * cm_per_mpc)**3 * self.cosm.nH(z) / 3.
            Nlya = (zeta_lya / zeta) * Nlyc
            
            # This is defined to guarantee that the integrated Nalpha history
            # recovers the total number.
            tmax = Ri / c / s_per_myr
            t0 = self.cosm.t_of_z(50.) / s_per_myr
            t = self.cosm.LookbackTime(z, 50.) / s_per_myr
            N0 = Nlya / (np.exp(tmax / t0) - 1.)
            Ndot = lambda tt: N0 * np.exp(tt / t0)
            
            Nofr = lambda r: Ndot(t - r * s_per_myr / cm_per_mpc / c) \
                / 4. / np.pi / (r * cm_per_mpc)**2 / c
            
            nH = self.cosm.nH(z)
            r_p = fsolve(lambda rr: Nofr(rr) - nH, 1.)
                    
            r_c = r_p * (1. + z)

        else:
            raise NotImplemented('help')

        return Rc
        
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
            #print("Loaded ps_{} at z={} from cache.".format(term, z))
            return self._cache_ps_[z][term]        
    
    def _cache_Vo(self, sep, Ri, Rh, Rc):
        if not hasattr(self, '_cache_Vo_'):
            self._cache_Vo_ = {}

        if sep not in self._cache_Vo_:
            self._cache_Vo_[sep] = self.overlap_volumes(sep, Ri, Rh, Rc)

        return self._cache_Vo_[sep]
        
    def _cache_p(self, z, term):
        if not hasattr(self, '_cache_p_'):
            self._cache_p_ = {}
    
        if z not in self._cache_p_:
            self._cache_p_[z] = {}
    
        if term not in self._cache_p_[z]:
            return None
        else:
            return self._cache_p_[z][term]
    
    def ExpectationValue(self, z, zeta, term='i', Rh=0.0, Th=500.0, Ts=None):
        """
        Compute the probability that a point is something.
        
        These are the one point terms in brackets, e.g., <x>, <x delta>, etc.
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
            val = 1. - self.ExpectationValue(z, zeta, term='i', Rh=Rh)
        elif term == 'h':
            assert Rh is not None
            Qh = self.BubbleShellFillingFactor(z, zeta, Rh) 
            val = Qh
        elif term.strip() == 'i*h':
            assert Rh is not None
            Qi = self.MeanIonizedFraction(z, zeta)
            Qh = self.BubbleShellFillingFactor(z, zeta, Rh) 
            val = Qh * Qi # implicit * 1
        elif term.strip() == 'i*c':
            assert Rh is not None
            c = self.TempToContrast(z, Th, Ts)
            Qi = self.MeanIonizedFraction(z, zeta)
            Qh = self.BubbleShellFillingFactor(z, zeta, Rh) 
            val = Qh * Qi * c
        elif term == 'c':
            Qh = self.BubbleShellFillingFactor(z, zeta, Rh)
            c = self.TempToContrast(z, Th, Ts)            
            val = c * Qh
        elif term == 'i*d':
            if self.pf['ps_include_density_xcorr']:
                avg_xi = self.ExpectationValue(z, zeta, term='i')
                val = avg_xi * self._B0(z, zeta)
            else:
                val = 0.0
        elif term == 'n*d':
            val = -self.ExpectationValue(z, zeta, term='i*d')
        elif term == 'n*c':
            c = self.TempToContrast(z, Th, Ts)
            avg_x = self.ExpectationValue(z, zeta, term='n')
            val = avg_x * c
        elif term == 'n*d*c':
            if self.pf['ps_include_density_xcorr']:
                raise NotImplemented('help')    
            else:
                val = 0.0
        elif term == 'psi':
            # <psi> = <x (1 + d)> = <x> + <xd> = 1 - <x_i> + <d> - <x_i d>
            #       = 1 - <x_i> - <x_i d>
            
            avg_xd = self.ExpectationValue(z, zeta, term='n*d')            
            val = self.ExpectationValue(z, zeta, term='n') + avg_xd

        elif term == 'phi':
            # <phi> = <psi * (1 + c)> = <psi> + <psi * c>
            # <phi * c> = <x * c> + <x * c * d>
            
            avg_psi = self.ExpectationValue(z, zeta, term='psi')
            
            avg_xcd = self.ExpectationValue(z, zeta, term='n*d*c')            
            avg_xc = self.ExpectationValue(z, zeta, term='n*c')
            avg_psi_c = avg_xc + avg_xcd
                            
            val = avg_psi + avg_psi_c
                    
        else:
            raise ValueError('Don\' know how to handle <{}>'.format(term))
        
        self._cache_p_[z][term] = val
        
        return val
        
    def JointProbability(self, z, zeta, R, term='ii', Rh=0.0, R3=0.0, 
        Th=500.0, Ts=None):
        """
        Compute the joint probability that two points are ionized, heated, etc.

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
        
        if term[0] != term[1]:
            is_xcorr = True
        else:
            is_xcorr = False
            
        if self.pf['bubble_size_dist'].lower() != 'fzh04':
            raise NotImplemented('Only know fzh04 BSD so far!')

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
        xibar = Q = self.MeanIonizedFraction(z, zeta)
        
        Rzeros = np.zeros_like(R)
        
        # If reionization is over, don't waste our time!
        if xibar == 1:
            return np.ones(R.size), Rzeros, Rzeros
                
        ##
        # Check for derived quantities like psi, phi
        ##
        if term == 'psi':
            # <psi psi'> = <x (1 + d) x' (1 + d')> = <xx'(1+d)(1+d')>
            #            = <xx'(1 + d + d' + dd')>
            #            = <xx'> + 2<xx'd> + <xx'dd'>
            
            xx, xx1, xx2 = self.JointProbability(z, zeta, R=R, term='nn')
            dd = self.JointProbability(z, zeta, R, term='dd')
            
            if self.pf['ps_include_density_xcorr']:
                avg_x = self.ExpectationValue(z, zeta, term='n')
                xd, xd1, xd2 = self.JointProbability(z, zeta, R=R, term='nd')
                xxd = avg_x * xd
            else:
                xd = xxd = 0.0
            
            avg_xd = self.ExpectationValue(z, zeta, term='n*d')
            xxdd, xxdd1, xxdd2 = self.JointProbability(z, zeta, R=R, term='xxdd')
            
            jp = xx + 2 * xxd + xxdd
            jp1 = np.zeros_like(xx1)
            jp2 = np.zeros_like(xx)
            
            return jp, jp1, jp2
                
        elif term == 'phi':
            Phi, junk1, junk2 = self.JointProbability(z, zeta, R, term='Phi',
                Rh=Rh, Ts=Ts, Th=Th)
            jp_psi, jp_psi1, jp_psi2 = self.JointProbability(z, zeta, R,
                term='psi', Rh=Rh, Ts=Ts, Th=Th)
            avg_psi = self.ExpectationValue(z, zeta, term='psi')
            avg_phi = self.ExpectationValue(z, zeta, term='phi', 
                Rh=Rh, Ts=Ts, Th=Th)

            return jp_psi + Phi - (avg_phi**2 - avg_psi**2), \
                Rzeros, Rzeros

        elif term == 'Phi':
            avg_c = self.ExpectationValue(z, zeta, term='c', 
                Rh=Rh, Ts=Ts, Th=Th)

            jp_xxc, jp_xxc1, jp_xxc2 = \
                self.JointProbability(z, zeta, R, term='xxc',
                Rh=Rh, Ts=Ts, Th=Th)
            jp_xxcc, jp_xxcc1, jp_xxcc2 = \
                self.JointProbability(z, zeta, R, term='xxcc',
                Rh=Rh, Ts=Ts, Th=Th)
        
            Phi = 2. * (jp_xxc + jp_xxcc)
        
            return Phi, Rzeros, Rzeros        
            
        elif term == 'cc':
            jp_hh, jp_hh1, jp_hh2 = \
                self.JointProbability(z, zeta, R, term='hh',
                Rh=Rh, Ts=Ts, Th=Th)
            c = self.TempToContrast(z, Th, Ts)
            
            return jp_hh * c**2, jp_hh1 * c**2, jp_hh2 * c**2    
          
        elif term == 'ic':
            jp_ih, jp_ih1, jp_ih2 = \
                self.JointProbability(z, zeta, R, term='ih',
                Rh=Rh, Ts=Ts, Th=Th)
            c = self.TempToContrast(z, Th, Ts)
            
            jp_ih = jp_ih1 = jp_ih2 = Rzeros
            
            return jp_ih * c, jp_ih1 * c, jp_ih2 * c  
                
        elif term == 'xxc':
            #c = self.TempToContrast(z, Th, Ts)
            avg_xc = self.ExpectationValue(z, zeta, term='n*c', 
                Rh=Rh, Ts=Ts, Th=Th)
            jp_ic, jp_ic1, jp_ic2 = \
                self.JointProbability(z, zeta, R, term='ic',
                Rh=Rh, Ts=Ts, Th=Th)
            
            return avg_xc - jp_ic, Rzeros, Rzeros
            
        elif term == 'xxcc':
            xx, xx1, xx2 = self.JointProbability(z, zeta, R, term='nn')
            cc, cc1, cc2 = self.JointProbability(z, zeta, R, term='cc')
            
            c = self.TempToContrast(z, Th, Ts)
            avg_x = self.ExpectationValue(z, zeta, term='n')
            avg_c = self.ExpectationValue(z, zeta, term='c')
            
            #if self.pf['ps_include_density_xcorr']:
            xc, xc1, xc2 = self.JointProbability(z, zeta, R, term='nc')
            avg_xc = self.ExpectationValue(z, zeta, term='n*c')
            #else:
            #    xc = avg_xc = 0.0
            
            if self.pf['ps_use_wick']:        
                xxcc = xx * cc + 2 * avg_xc + xc**2
            else:
                # Use binary model.            
                xxcc = cc
                
            return xxcc, Rzeros, Rzeros
            
        elif term == 'xxdd':
            xx, xx1, xx2 = self.JointProbability(z, zeta, R, term='nn')
            dd = self.JointProbability(z, zeta, R, term='dd')
            avg_x = self.ExpectationValue(z, zeta, term='n')
            
            if self.pf['ps_use_wick']:
                xd, xd1, xd2 = self.JointProbability(z, zeta, R, term='nd')
                avg_xd = self.ExpectationValue(z, zeta, term='n*d')
                xxdd = xx * dd + 2 * avg_xd + xd**2
                
                return xxdd, Rzeros, Rzeros
            else:
                # Use binary model.            
                return dd * avg_x**2, Rzeros, Rzeros
                
        elif term == 'dd':
            # Equivalent to correlation function since <d> = 0
            return self.spline_cf_mm(z)(np.log(R))#, np.zeros_like(R), np.zeros_like(R)
        elif term == 'nd':
            idt, id1, id2 = self.JointProbability(z, zeta, R, term='id')
            return -idt, -id1, -id2
        elif term == 'nn':                                                         
            ii, ii1, ii2 = self.JointProbability(z, zeta, R, term='ii')
            return 1. - 2. * Q + ii, 1. - 2. * Q + ii1, 1. - 2. * Q + ii2
        elif term == 'nc':  
            avg_c = self.ExpectationValue(z, zeta, term='c')                                                       
            ic, ic1, ic2 = self.JointProbability(z, zeta, R, term='ic')
            return avg_c - ic, avg_c - ic1, avg_c - ic2
            
        # Some stuff we need
        Ri, Mi, dndm = self.BubbleSizeDistribution(z, zeta)
        
        Qh = self.BubbleShellFillingFactor(z, zeta, Rh)
                
        # Minimum bubble size            
        Mmin = self.Mmin(z) * zeta
        iM = np.argmin(np.abs(Mi - Mmin))
        
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
        
        # Loop over scales
        B1 = np.zeros(R.size)
        B2 = np.zeros(R.size)
        BT = np.zeros(R.size)
        for i, sep in enumerate(R):
            
            ##
            # Note: each element of this loop we're constructing an array
            # over bubble mass, which we then integrate over to get a total
            # probability. The shape of every quantity should be `self.m`.
            ##
            
            
            # Could do this once externally, i.e., not for each term.
            
            # Yields: V11, V12, V13, V22, V23, V33            
            # Remember: these radii arrays depend on redshift (through delta_B)
            
            all_V = self.overlap_volumes(sep, Ri, Rh, R3)
        
            # For two-halo terms, need bias of sources.
            if self.pf['ps_include_bias']:
                ep = self.excess_probability(z, zeta, sep, term)
                #elif term == 'hh':
                #    # Undo normalization by ionized volume filling factor!
                #    ep = self.excess_probability(z, zeta, sep, term) #* Q / Qh
                #else:
                #    ep = np.zeros_like(self.m)
                    #raise NotImplemented('help')    
            else:
                ep = np.zeros_like(self.m)

            # Correction factor for two-halo term. Occassionally must
            # be adapted which is why we introduce it here, rather than 
            # much lower.
            corr = (1. + ep)

            ##
            # For each zone, figure out volume of region where a
            # single source can ionize/heat/couple both points, as well
            # as the region where a single source is not enough (Vss_ne)
            ##
            if term == 'ii':
                Vo = all_V[0]
                Vi = 4. * np.pi * Ri**3 / 3.
            
                Vss_ne_1 = Vi - Vo
                Vss_ne_2 = Vss_ne_1
            
            elif term == 'hh':
                
                #if self.pf['powspec_temp_method'] == 'xset':
                #    # In this case, zeta == zeta_X, and Ri is really the radius
                #    # of the heated region
                #    Vo = all_V[0]
                #    
                #    Vss_ne_1 = 4. * np.pi * Ri**3 / 3.
                #    Vss_ne_2 = Vss_ne_1
                #                    
                #    limiter = None
                #elif self.pf['powspec_temp_method'] == 'shell':    
                
                Vii = all_V[0]
                _integrand1 = dndm * Vii
                
                _exp_int1 = np.exp(-simps(_integrand1[iM:] * Mi[iM:],
                    x=np.log(Mi[iM:])))
                _P1 = (1. - _exp_int1)
                
                # Region in which two points are heated by the same source
                Vo = all_V[3]
                
                # These are functions of mass
                Vh = 4. * np.pi * (Rh - Ri)**3 / 3.
                                
                
                                
                # Subtract off region of the intersection HH volume
                # in which source 1 would do *anything* to point 2.
                #Vss_ne_1 = Vh - (Vo - self.IV(sep, Ri, Rh) + all_V[0])
                Vss_ne_1 = Vh - Vo #- all_V[0] - all_V[1]
                 
                #Vss_ne_1 = Vh - Vo
                Vss_ne_2 = Vss_ne_1 
                                                
                # At this point, single source might heat one pt ionize other
                
                #self.IV(sep, Rh, Rc)
                
                
                #Vss_ne_1 = 4. * np.pi * (Rh - Ri)**3 / 3. \
                #         - 0.5 * all_V[1] - all_V[0]
                # Why the factor of 1/2? all_V[0] needed?
                
                integrand1 = dndm * Vo
                
                # Undo BSD kludge
                #integrand1 *= np.log(1. - Qh) / Q
                
                

                exp_int1 = np.exp(-simps(integrand1[iM:] * Mi[iM:],
                    x=np.log(Mi[iM:])))
                P1 = (1. - exp_int1)
                #P1 = simps(integrand1[iM:] * Mi[iM:], x=np.log(Mi[iM:]))

                #print(sep, P1 / P1_2)

                # Start chugging along on two-bubble term   
                if np.any(Vss_ne_1 < 0):
                    N = sum(Vss_ne_1 < 0)
                    print('R={}: Vss_ne_1 (hh) < 0 {} / {} times'.format(sep, N, len(Rh)))

                integrand2_1 = dndm * np.abs(Vss_ne_1)
                integrand2_2 = dndm * np.abs(Vss_ne_2)

                int1 = simps(integrand2_1[iM:] * Mi[iM:], 
                    x=np.log(Mi[iM:]))        
                int2 = simps(integrand2_2[iM:] * Mi[iM:] * corr[iM:], 
                    x=np.log(Mi[iM:]))

                exp_int2 = np.exp(-int1)
                exp_int2_ex = np.exp(-int2)
                
                P2 = (1. - exp_int2) * (1. - exp_int2_ex) * (1. - P1)

                B1[i] = P1
                B2[i] = min(P2, Qh**2)
                # Must correct for the fact that Qi+Qh<=1
                         
                continue    

            elif term == 'id':
                
                if not self.pf['ps_include_density_xcorr']:
                    B1[i] = B2[i] = 0
                    continue
                    
                Vo = all_V[0]
                
                delta_B = self._B(z, zeta)
                
                # Multiply by Mi just to do the integral in log
                _P1_integrand = dndm * Vo * (1. + delta_B)
                #P1 = np.trapz(_P1_integrand[iM:], x=np.log(Mi[iM:]))
                
                exp_int1 = np.exp(-simps(_P1_integrand[iM:] * Mi[iM:],
                    x=np.log(Mi[iM:])))
                P1 = (1. - exp_int1)
                
                #Vss_ne_1 = Vss_ne_2 = np.zeros_like(Ri)
        
                bh = self.halos.Bias(z)
                bb = self.bubble_bias(z, zeta)
                xi = self.spline_cf_mm(z)(np.log(sep))
                
                Vi = 4. * np.pi * Ri**3 / 3.
                
                Vss_ne_1 = Vi - Vo
                Vss_ne_2 = Vss_ne_1
                
                
                #dndm_h = self.halos.tab_dndm[iz_hmf]
                #iM_h = np.argmin(np.abs(Mi - self.Mmin(z)))
                
                _prob_i_d = dndm * bb * xi_dd[i] * Vi * Mi
                prob_i_d = simps(_prob_i_d[iM:], x=np.log(Mi[iM:]))
                
                #_Pout = bh * dndm_h
                #Pout = np.trapz(_Pout[iM_h:], x=np.log(self.halos.M[iM_h:]))
                #_Pout_int = 
                #Pout = np.trapz(_Pout_int[iM:], x=np.log(Mi[iM:]))
                Pout = prob_i_d
                
                #print(z, i, sep, Pin, Pout)
                
                B1[i] = P1
                B2[i] = Pout
                
                continue
                                                                    
            elif term == 'cc':
                
                #if self.pf['powspec_lya_method'] > 0:
                #    Vo = self.IV(sep, Rc, Rc)
                #    Vss_ne_1 = 4. * np.pi * Rc**3 / 3. - Vo
                #    Vss_ne_2 = Vss_ne_1
                #else:
                Vo = all_V[-1]
                Vss_ne_1 = 4. * np.pi * (R3 - Rh)**3 / 3. \
                         - (self.IV(sep, R3, R3) - self.IV(sep, Rh, R3))
                Vss_ne_2 = Vss_ne_1
            
                
                #if self.pf['bubble_pod_size_func'] in [None, 'const', 'linear']:
                #    Vo_sh_r1, Vo_sh_r2, Vo_sh_r3 = \
                #        self.overlap_region_shell(sep, np.maximum(Ri, Rh), Rc)
                #    
                #    Vo = Vo_sh_r1 - 2. * Vo_sh_r2 + Vo_sh_r3
                #    
                #    Vss_ne = 4. * np.pi * (Rc - np.maximum(Ri, Rh))**3 / 3. - Vo
                #elif self.pf['bubble_pod_size_func'] == 'approx_sfh':
                #    raise NotImplemented('sorry!')
                #elif self.pf['bubble_pod_size_func'] == 'fzh04':
                #    Vo = self.overlap_region_sphere(sep, Rc)
                #    Vss_ne = 4. * np.pi * Rc**3 / 3. - Vo
                #else:
                #    raise NotImplemented('sorry!')
        
                limiter = None
        
            #elif term == 'hc':
            #    Vo = all_V[-2]
            #
            #    # One point in cold shell of one bubble, another in
            #    # heated shell of completely separate bubble.
            #    # Need to be a little careful here!
            #    Vss_ne_1 = 4. * np.pi * (Rh - Ri)**3 / 3. \
            #             - self.IV(sep, Rh, Rc)
            #    Vss_ne_2 = 4. * np.pi * (Rc - Rh)**3 / 3. \
            #             - (self.IV(sep, Rc, Rc) - self.IV(sep, Rh, Rc)) \
            #             - self.IV(sep, Rh, Rc)
            #
            #    limiter = None

            elif term == 'ih':
                #Vo_sh_r1, Vo_sh_r2, Vo_sh_r3 = \
                #    self.overlap_region_shell(sep, Ri, Rh)
                #Vo = 2. * Vo_sh_r2 - Vo_sh_r3
                Vo = all_V[1]
                
                Vi = 4. * np.pi * Ri**3 / 3.
                Vh = 4. * np.pi * (Rh - Ri)**3 / 3.

                # Volume in which I ionize but don't heat (or ionize) the other pt.
                Vss_ne_1 = Vi - self.IV(sep, Ri, Rh)
                         
                # Volume in which I heat but don't ionize (or heat) the other pt.
                # Same as hh term?
                Vss_ne_2 = Vh - self.IV(sep, Ri, Rh)
        
                if np.any(Vss_ne_2 < 0):
                    N = sum(Vss_ne_2 < 0)
                    print('R={}: Vss_ne_2 (ih) < 0 {} / {} times'.format(sep, N, len(Rh)))
            
                
            #elif term == 'ic':
            #    Vo = all_V[2]
            #    Vss_ne_1 = 4. * np.pi * Ri**3 / 3. \
            #             - (self.IV(sep, Ri, Rc) - self.IV(sep, Ri, Rh))
            #    Vss_ne_2 = 4. * np.pi * (Rc - Rh)**3 / 3. \
            #             - (self.IV(sep, Ri, Rc) - self.IV(sep, Ri, Rh))
            
            elif term == 'hd':
                Vo = all_V[3]
                
                delta_B = self._B(z, zeta, zeta)
                _Pin_int = dndm * Vo * Mi * delta_B
                Pin = np.trapz(_Pin_int[iM:], x=np.log(Mi[iM:]))
                
                Vss_ne_1 = Vss_ne_2 = np.zeros_like(Ri)
                            
                #xi_dd = data['xi_dd_c'][i]
                #
                #bHII = self.bubble_bias(z, zeta)
                                
                bh = self.halos.Bias(z)
                bb = self.bubble_bias(z, zeta)
                
                Vi = 4. * np.pi * (Rh - Ri)**3 / 3.
                _prob_i_d = dndm * bb * xi_dd[i] * Vi * Mi
                prob_i_d = np.trapz(_prob_i_d[iM:], x=np.log(Mi[iM:]))
                
                Pout = prob_i_d #+ data['Qh']
                
                print(z, i, sep, Pin, Pout)
                
                limiter = None
                
            else:
                print('Skipping %s term for now' % term)
                #raise NotImplemented('under construction')
                break
        
        
            ###
            ## Currently, only make it here for auto- terms
            ###
        
            # Compute the one-bubble term
            integrand1 = dndm * Vo
                     
            exp_int1 = np.exp(-simps(integrand1[iM:] * Mi[iM:],
                x=np.log(Mi[iM:])))
            P1 = (1. - exp_int1)
            #P1 = simps(integrand1[iM:] * Mi[iM:], x=np.log(Mi[iM:]))
            
            #print(sep, P1 / P1_2)
                                
            # Start chugging along on two-bubble term   
            
            #if np.any(Vss_ne_1 < 0):
            #    print('Vss_ne_1 ={} at R={}'.format(Vss_ne_1, sep))
            ##else:
            ##    print('Vss_ne_1 =OK at R={}'.format(sep))
            #if np.any(Vss_ne_2 < 0):     
            #    print('Vss_ne_2 < 0 at', sep)
                             
            integrand2_1 = dndm * Vss_ne_1
            integrand2_2 = dndm * Vss_ne_2
                    
            int1 = simps(integrand2_1[iM:] * Mi[iM:], 
                x=np.log(Mi[iM:]))        
            int2 = simps(integrand2_2[iM:] * Mi[iM:] * corr[iM:], 
                x=np.log(Mi[iM:]))
                    
            exp_int2 = np.exp(-int1)
            exp_int2_ex = np.exp(-int2)
            
             
            P2 = (1. - exp_int2) * (1. - exp_int2_ex) * (1. - P1)
                        
            B1[i] = P1
            B2[i] = P2            
            
        # Replace two-halo term with a power-law all the way down?
        BT = B1 + B2
       
        self._cache_jp_[z][term] = R, BT, B1, B2
        
        return BT, B1, B2


    def CorrelationFunction(self, z, zeta=None, R=None, term='ii', 
        Rh=0.0, R3=0.0, Th=500., Tc=1., Ts=None,
        assume_saturated=True, include_xcorr=False, include_ion=True,
        include_temp=False, include_lya=False, include_density=True,
        include_21cm=True):
        """
        Compute the correlation function of some general term.
        
        """
        
        Q = self.MeanIonizedFraction(z, zeta)
        Qh = self.BubbleShellFillingFactor(z, zeta, Rh)
        
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
        # Matter correlation function -- we have this tabulated already.
        ##
        if term == 'mm':
            
            if not include_density:
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
            jp_ii, jp_ii_1, jp_ii_2 = \
                self.JointProbability(z, zeta, R=R, term='ii')
                                
            # Add optional correction to ensure limiting behavior?        
            if self.pf['ps_volfix']:
                if Q < 0.5:
                    ev_ii = jp_ii_1 + jp_ii_2
                else:
                    ev_ii = (1. - Q) * jp_ii_1 + Q**2
            else:    
                ev_ii = jp_ii
    
            ev_i = self.ExpectationValue(z, zeta, term='i')
            cf = ev_ii - ev_i**2
            
        ##
        # Ionization-density cross correlation function
        ##
        elif term == 'id':
            #raise NotImplemented('fix me please')
            jp_ii, jp_ii_1, jp_ii_2 = \
                self.JointProbability(z, zeta, R, Q, term='ii')

            jp_im, jp_im_1, jp_im_2 = \
                self.JointProbability(z, zeta, R, Q, term='id')
        
            # Add optional correction to ensure limiting behavior?        
            if self.pf['ps_volfix']:
                if Q < 0.5:
                    ev = jp_1h + jp_2h
                else:
                    ev = jp_1h #- jp_ii_1
            else:    
                ev = jp
        
            # Equivalent to correlation function in this case.
            cf = ev - Q * np.min(self._B0(z, zeta))

        ##
        # Temperature correlation function
        ##
        elif term == 'hh':
            jp_hh, jp_hh_1, jp_hh_2 = \
                self.JointProbability(z, zeta, R=R, term='hh', Rh=Rh)
            
            if self.pf['ps_volfix']:
                if (Qh < 0.5) and (Q < 0.5):
                    ev_hh = jp_hh_1 + jp_hh_2
                else:
                    # Should this 1-Qh factor be 1-Qh-Qi?
                    ev_hh = (1. - Qh) * jp_hh_1 + Qh**2
            else:    
                ev_hh = jp_hh 
            
            ev_h = self.ExpectationValue(z, zeta, term='h')
            cf = ev_hh - h**2
            
        # c = contrast, instead of 'c' for cold, use '3' for zone 3 (later)    
        elif term == 'cc':
            jp_cc, jp_cc_1, jp_cc_2 = \
                self.JointProbability(z, zeta, R=R, term='cc', Rh=Rh)
            
            c = self.TempToContrast(z, Th, Ts)
            
            ev_c = self.ExpectationValue(z, zeta, term='c')
            
            #else:
            #    # Remember, this is for the hot/cold term
            #    Csq = (Tcmb / (Tk - Tcmb)) * delta_T[0] * delta_T[1] \
            #        / (1. + delta_T[0]) / (1. + delta_T[1])
            #    C = np.sqrt(C)    
            
                
            # Add optional correction to ensure limiting behavior?        
            if self.pf['ps_volfix']:
                if (Qh < 0.5) and (Q < 0.5):
                    ev_cc = (jp_cc_1 + jp_cc_2)
                else:
                    ev_hh = (1. - Qh) * jp_cc_1 + ev_c**2
            else:    
                ev_cc = jp_cc
                           
            cf = ev_cc - ev_c**2
            
        ##
        # Ionization-heating cross-correlation function
        ##    
        elif term == 'ih':
            jp, jp_1, jp_2 = \
                self.JointProbability(z, zeta, R=R, term='ih', Rh=Rh)
        
            C = self.TempToContrast(z, Th, Ts)
        
            #else:
            #    # Remember, this is for the hot/cold term
            #    Csq = (Tcmb / (Tk - Tcmb)) * delta_T[0] * delta_T[1] \
            #        / (1. + delta_T[0]) / (1. + delta_T[1])
            #    C = np.sqrt(C)    
        
        
            # Add optional correction to ensure limiting behavior?        
            if self.pf['ps_volfix']:
                if (Qh < 0.5) and (Q < 0.5):
                    ev = (jp_1 + jp_2) * C
                else:
                    ev = (1. - Qh) * jp_1 * C + Qh * Q * C
            else:    
                ev = jp * C + Qh * Q * C
        
            #ev_hh = jp_hh               
            #                        
            #cf = ev_hh - Qh**2
        
            cf = ev - Q * C * Qh
        
        
        ##
        # 21-cm correlation function
        ##  
        elif term in ['21', 'phi', 'psi']:
            
            ev_2pt = self.JointProbability(z, zeta, R=R, term='psi')
            ev_1pt = self.ExpectationValue(z, zeta, term='psi')
            
            cf_psi = ev_2pt - ev_1pt**2
            
            # Ionization-ionization correlation function
            #xi_ii = self.CorrelationFunction(z, zeta, R=R, term='ii')
            #
            ## Correlation function in neutral fraction same by linearity.
            #xi_nn = xi_ii
            #
            ## Matter correlation function    
            #xi_mm = self.CorrelationFunction(z, zeta, R=R, term='mm')
            #    
            #if include_xcorr:
            #    #raise NotImplemented('help')
            #    xi_im = self.CorrelationFunction(z, zeta, R=R, term='id')
            #    xi_nm = -xi_im    
            #else:
            #    xi_im = xi_nm = 0.0
            #
            #xbar = (1. - Q)
            #
            ## This is the solution in the saturated limit.
            ## If the correlations vanish, e.g., on large scales, we should
            ## recover the matter power CF * xbar**2            
            #xi_psi = xi_nn * (1. + xi_mm) + xbar**2 * xi_mm \
            #       + xi_nm * (xi_nm + 2. * xbar)
            #
            ##
            # Temperature fluctuations
            ##
            if (include_temp or include_lya) and term in ['phi', '21']:
                
                Phi = self.JointProbabilty(z, zeta, R=R, term='Phi',
                    Rh=Rh, Ts=Ts, Th=Th)
                avg_psi = self.ExpectationValue(z, zeta, term='psi')
                avg_phi = self.ExpectationValue(z, zeta, term='phi', 
                    Rh=Rh, Ts=Ts, Th=Th)        
                    
                cf_21 = cf_phi = cf_psi + Phi - (avg_phi - avg_psi**2)
                
                ##
                # <phi phi'> = a lot
                # <phi> <phi'> = <psi> <psi'> + 2 <psi><psi C> + <psi C>^2
                ##
            
                ##
                # Gather Poisson term first.
                ##
            
                # Mean contrast of heated regions
                #avg_x = self.ExpectationValue(z, zeta, term='n')
                #avg_C = self.ExpectationValue(z, zeta, term='c', Rh=Rh,
                #    Th=Th, Ts=Ts)
                #
                #avg_psi = avg_x # neglecting <x_i d> right now
                #
                #avg_del_C = 0.0#self.ExpectationValue(z, zeta, term='Cd')
                #avg_psi_C = avg_C + avg_del_C
                #                
                #avg_phi_sq = avg_psi**2 + 2 * avg_psi * avg_psi_C + avg_psi_C**2
                #
                ###
                ## On to the two point terms
                ###
                #
                ## Hot-hot correlation function
                #xi_hh = self.CorrelationFunction(z, zeta, R=R, term='hh', 
                #    Rh=Rh, Ts=Ts, Th=Th)
                #
                ## Ionized-heated correlation function
                #xi_ih = self.CorrelationFunction(z, zeta, R=R, term='ih', 
                #    Rh=Rh, Ts=Ts, Th=Th)    
                #
                ## Convert to neutral-heated correlation function
                ## <xC'> = <(1-x_i)C'> = <C'> - <x_i C'> 
                ##       = Cbar - (xi_ih + Q * Cbar)
                ##xi_nh = Cbar - (xi_ih + Cbar * Q)
                #xxC = avg_C - xi_ih
                
                
                # On large scales, this is Cbar * xbar...uh oh
                
                # Should <C> = 0?
            
                # The two easiest terms in the unsaturated limit are those
                # not involving the density, <x x' C'> and <x x' C C'>.
                # Under the binary field(s) approach, we can just write
                # each of these terms down
                
                # This is the <x x' C C'> term, assumes fields are ~Gaussian.
                # My (current) Eq. 47
                #xxCC = 2 * ((xi_nn + xbar**2) * (xi_hh + Cbar**2) \
                #       + xbar**2 + (xi_nh + xbar**2 * Cbar**2))
                #xxCC = xi_hh + avg_C**2       
                #       
                ## On large scales, this is 
                ## = 2 * (xbar**2 * Cbar**2 + xbar**2 + xbar * Cbar + xbar**2 * Cbar**2)
                #
                ## The <xx'C'> term is just <x><C>
                ##xxC = avg_x * avg_C
                #
                #Phi = xxCC + xxC #\
                    #- 2 * (xbar**2 * Cbar**2 + xbar**2 + xbar * Cbar + xbar**2 * Cbar**2)
                
                # In R -> inf limit, we recover:
                # 2 * (xbar**2 * Cbar**2 + xbar**2 + xbar**2 * Cbar**2 + xbar * Cbar)

                
                # Could include more cross terms...

                # Need to make sure this doesn't get saved at native resolution!
                #data['phi_u'] = phi_u
                
                

                #xi_phi = xi_psi + Phi - (avg_phi_sq - avg_psi**2)
                
                # Kludge to guarantee convergence
                #xi_21 = xi_phi #- Phi[-1]
            else:
                cf_21 = cf_psi

            cf = cf_21

        else:
            raise NotImplemented('help')
        
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
        
