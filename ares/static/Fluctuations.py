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
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from ..physics.HaloModel import HaloModel
from ..util.Math import LinearNDInterpolator
from ..populations.Composite import CompositePopulation
from ..physics.CrossSections import PhotoIonizationCrossSection
from ..physics.Constants import g_per_msun, cm_per_mpc, dnu, s_per_yr, c, \
    s_per_myr, erg_per_ev, k_B, m_p, dnu

root2 = np.sqrt(2.)

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
    
    def overlap_volumes(self, dr, R1, R2, R3):
        """
        This will yield 6 terms.
    
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
    
    def BubbleShellFillingFactor(self, z, zeta):
        if self.pf['bubble_size_dist'] is None:
            R_b = self.pf['bubble_size']
            V_b = 4. * np.pi * R_b**3 / 3.
            n_b = self.BubbleDensity(z)
        
            Qh = 1. - np.exp(-n_b * V_b)
        elif self.pf['bubble_size_dist'].lower() == 'fzh04':
            Ri, Mi, dndm = self.BubbleSizeDistribution(z, zeta)

            Rh, Rc = self.BubbleShellRadius(z, Ri)
            
            if np.logical_and(np.all(Rh == 0), np.all(Rc == 0)):
                return 0., 0.
            
            Mmin = self.Mmin(z)
            iM = np.argmin(np.abs(Mmin * zeta - Mi))

            Vi = 4. * np.pi * Ri**3 / 3.   
            Vsh1 = 4. * np.pi * (Rh - Ri)**3 / 3.
            
            dndlnm = dndm * Mi
            
            if np.all(Rc == 0):
                Vsh2 = Qc = 0.0
            else:
                Vsh2 = 4. * np.pi * (Rc - Rh)**3 / 3.
                Qc = np.trapz(dndlnm[iM:] * Vsh2[iM:], x=np.log(Mi[iM:]))
            
            Qi = np.trapz(dndlnm[iM:] * Vi[iM:], x=np.log(Mi[iM:]))
            Qh = np.trapz(dndlnm[iM:] * Vsh1[iM:], x=np.log(Mi[iM:]))
            
            #if self.pf['powspec_rescale_Qion'] and self.pf['powspec_rescale_Qhot']:
            #    norm = min(zeta * self.halos.fcoll_2d(z, np.log10(self.Mmin(z))), 1)
            #    
            #    corr = (norm / Qi)
            #    Qh *= corr

        else:
            raise NotImplemented('Uncrecognized option for BSD.')
         
        return min(Qh, 1.), min(Qc, 1.)

    @property
    def bsd_model(self):
        return self.pf['bubble_size_dist'].lower()

    def BubbleFillingFactor(self, z, zeta, rescale=False):
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

            if rescale:
                Qi = zeta * self.halos.fcoll_2d(z, logM)
            else:
                
                # Mi should just be self.m? No.
                Ri, Mi, dndm = self.BubbleSizeDistribution(z, zeta, rescale)
                Vi = 4. * np.pi * Ri**3 / 3.

                dndlnm = dndm * Mi
                
                iM = np.argmin(np.abs(Mmin * zeta - Mi))
                
                Qi = np.trapz(dndlnm[iM:] * Vi[iM:], x=np.log(Mi[iM:]))

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

    def mean_bubble_bias(self, z, zeta, term='ii', Q=None, rescale=False):
        """
        Note that we haven't yet divided by QHII!
        """
        
        Ri, Mi, dndm = self.BubbleSizeDistribution(z, zeta, Q=Q, 
            rescale=rescale)
    
        if ('h' in term) or ('c' in term) and self.pf['powspec_temp_method'] == 'shell':
            Rh, Rc = self.BubbleShellRadius(z, Ri)
            R = Rh
        else:
            R = Ri
    
        V = 4. * np.pi * R**3 / 3.
    
        Mmin = self.Mmin(z) * zeta
        iM = np.argmin(np.abs(Mmin - self.m))    
        bHII = self.bubble_bias(z, zeta)
        
        #tmp = dndm[iM:]
        #print(z, len(tmp[np.isnan(tmp)]), len(bHII[np.isnan(bHII)]))
            
        #imax = int(min(np.argwhere(np.isnan(Ri))))
    
        return np.trapz(dndm[iM:] * V[iM:] * bHII[iM:] * Mi[iM:],
            x=np.log(Mi[iM:]))
    
    def spline_cf_mm(self, z):
        if not hasattr(self, '_spline_cf_mm_'):
            self._spline_cf_mm_ = {}
            
        if z not in self._spline_cf_mm_:
            iz = np.argmin(np.abs(z - self.halos.tab_z_ps))
            self._spline_cf_mm_[z] = interp1d(self.halos.tab_R, 
                self.halos.tab_cf_mm[iz], kind='cubic', bounds_error=False,
                fill_value=0.0)
        
        return self._spline_cf_mm_[z]    
    
    def excess_probability(self, z, zeta, R, Q=None, term='ii', rescale=False):
        """
        This is the excess probability that a point is ionized given that 
        we already know another point (at distance r) is ionized.
        """
        
        # For 'hh' term, should we modify R? Probably a small effect...
        
        if (Q is None) or rescale:
            Q = self.BubbleFillingFactor(z, zeta, rescale)
        
        # Function of bubble mass (bubble size)
        bHII = self.bubble_bias(z, zeta)
        bbar = self.mean_bubble_bias(z, zeta, term, Q, rescale) / Q
        
        if R < self.halos.tab_R.min():
            print("R too small")
        if R > self.halos.tab_R.max():
            print("R too big")    
        
        xi_dd = self.spline_cf_mm(z)(R)

        #if term == 'ii':
        return bHII * bbar * xi_dd 
        #elif term == 'im':
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
        
        iz = np.argmin(np.abs(z - self.halos.tab_z))
        D = self.halos.growth_factor[iz]

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


    def BubbleSizeDistribution(self, z, zeta, rescale=False, Q=None):
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
        
            # This is Eq. 9.38 from Steve's book.
            # The factors of 2, S, and Mi are from using dlns instead of 
            # dS (where S=s^2)
            dndm = rho0_m * self.pcross(z, zeta) * 2 * np.abs(self.dlns_dlnm) \
                * self.sigma**2 / Mi**2
                
            # Reionization is over!
            # Only use barrier condition if we haven't asked to rescale
            # or supplied Q ourselves.
            if (self._B0(z, zeta) <= 0) and (not rescale) and (Q is not None):
                reionization_over = True
                dndm = np.zeros_like(dndm)
            elif Q is not None:
                if Q == 1:
                    reionization_over = True
                    dndm = np.zeros_like(dndm)

        else:
            raise NotImplementedError('Unrecognized option: %s' % self.pf['bubble_size_dist'])
            
        if (not reionization_over):
            if (Q is None) or rescale:
                Q = self.BubbleFillingFactor(z, zeta, rescale=True)
            dndm *= -np.log(1. - Q) / Q
            
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
    
    def BubbleShellRadius(self, z, Ri, T1=None, T2=None):
        """
        Given a bubble radius (or array of them), convert to size of
        heated regions.
        """
        
        ##
        # If we made it here, we're doing something fancy.
        ##
        
        if not self.pf['ps_include_temp']:
            return np.zeros_like(Ri), np.zeros_like(Ri)

        if self.pf['powspec_temp_method'] == 'xset':
            return Ri, np.zeros_like(Ri)

        #if self.pf['bubble_shell_size_func'] == 'mfp':
        #
        #    sigma = PhotoIonizationCrossSection(self.Emin_X, species=0)
        #    lmfp_p = 1. / self.cosm.nH(z) / sigma / cm_per_mpc
        #    lmfp_c = lmfp_p * (1. + z)
        #    
        #    # Return value expected to be shell *radius*, not thickness, so
        #    # add bubble size back in.
        #    return Ri + lmfp_c
        #    
        #elif self.pf['bubble_shell_size_func'] == 'const':
        #    Nlyc = 4. * np.pi * (Ri * cm_per_mpc / (1. + z))**3 * self.cosm.nH(z) / 3.
        #    NX   = 1e-4 * Nlyc
        #    
        #    nrg_flux = NX * self.Emin_X * erg_per_ev
        #    Jx = nrg_flux * 0.2 
        #    
        #    #Rh = Ri \
        #    #   + (3 * Jx / self.cosm.adiabatic_cooling_rate(z) / 4. / np.pi)**(1./3.) \
        #    #   * (1. + z) / cm_per_mpc
        #
        #    return Rh

        #else:
        #    raise NotImplemented('help')

        # Just scale the heated region relative to the ionized region
        # in an even simpler way.

        to_ret = []
        for ii in range(2):
            
            if self.pf['bubble_shell_ktemp_zone_{}'.format(ii)] is None and \
               self.pf['bubble_shell_tpert_zone_{}'.format(ii)] is None:
                val = 0.0
            elif self.pf['bubble_shell_rsize_zone_{}'.format(ii)] is not None:
                val = Ri * (1. + self.pf['bubble_shell_rsize_zone_{}'.format(ii)])
            elif self.pf['bubble_shell_asize_zone_{}'.format(ii)] is not None:
                val = Ri + self.pf['bubble_shell_asize_zone_{}'.format(ii)]
            else:
                val = 0.0
                
            to_ret.append(val)
            
        return to_ret

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
            print("Loaded P_{} at z={} from cache.".format(term, z))
            return self._cache_jp_[z][term]
        
    def _cache_cf(self, z, term):
        if not hasattr(self, '_cache_cf_'):
            self._cache_cf_ = {}
    
        if z not in self._cache_cf_:
            self._cache_cf_[z] = {}
    
        if term not in self._cache_cf_[z]:
            return None
        else:
            print("Loaded xi_{} at z={} from cache.".format(term, z))
            return self._cache_cf_[z][term]    
    
    def _cache_ps(self, z, term):
        if not hasattr(self, '_cache_ps_'):
            self._cache_ps_ = {}
    
        if z not in self._cache_ps_:
            self._cache_ps_[z] = {}
    
        if term not in self._cache_ps_[z]:
            return None
        else:
            print("Loaded ps_{} at z={} from cache.".format(term, z))
            return self._cache_ps_[z][term]        
    
    def _cache_Vo(self, sep, Ri, Rh, Rc):
        if not hasattr(self, '_cache_Vo_'):
            self._cache_Vo_ = {}

        if sep not in self._cache_Vo_:
            self._cache_Vo_[sep] = self.overlap_volumes(sep, Ri, Rh, Rc)

        return self._cache_Vo_[sep]
    
    def JointProbability(self, z, zeta, R, Q=None, term='ii', 
        rescale=False, Rh=0.0, Rc=0.0):
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
            return np.interp(R, _R, _jp), np.interp(R, _R, _jp1), np.interp(R, _R, _jp2)
        
        # If volume filling factor not supplied, compute it.
        if Q is None:
            Q = self.BubbleFillingFactor(z, zeta, rescale)
            
        if Q == 1:
            return np.ones(R.size), np.zeros(R.size), np.ones(R.size)
        
        # Can convert to joint probability of the neutral fraction if we want.
        if term == 'nn':
            return 1. - 2. * Q + self.JointProbability(z, zeta, R, Q=Q, 
                term='ii', rescale=rescale)
        
        # Some stuff we need
        Ri, Mi, dndm = self.BubbleSizeDistribution(z, zeta, rescale, Q)
        
        if type(Rh) is np.ndarray:
            assert np.allclose(np.diff(Rh / Ri), 0.0), \
                "No support for absolute scaling of hot bubbles yet."
                        
            Qh = Q * (Rh[0] / Ri[0])**3
        else:
            Qh = 0.
        
        #Rh, Rc = self.BubbleShellRadius(z, Ri)
        #Rc = self.BubblePodRadius(z, Ri=Ri, zeta=zeta, 
        #    zeta_lya=zeta_lya)
                
        # Minimum bubble size            
        Mmin = self.Mmin(z) * zeta
        iM = np.argmin(np.abs(Mi - Mmin))
        
        iz = np.argmin(np.abs(z - self.halos.tab_z_ps))
        iz_hmf = np.argmin(np.abs(z - self.halos.tab_z))
        
        # Grab the matter power spectrum
        xi_dd = self.halos.tab_cf_mm[iz]
        
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
            
            all_V = self.overlap_volumes(sep, Ri, Rh, Rc)
        
            # For two-halo terms, need bias of sources.
            if self.pf['ps_include_bias']:
                if term == 'ii':
                    ep = self.excess_probability(z, zeta, sep, Q, term, rescale)
                elif term == 'hh':
                    ep = self.excess_probability(z, zeta, sep, Q, term, rescale)
                else:
                    ep = np.zeros_like(self.m)
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

                limiter = 'i'
            
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
                
                Vo = all_V[3]
                
                # Does this include single sources unable to affect the other
                # point at all? Yeah, should exclude anything but possible
                # hot-hot combo.
                Vh = 4. * np.pi * (Rh - Ri)**3 / 3.
                
                
                # Subtract off region of the intersection HH volume
                # in which source 1 would do *anything* to point 2.
                Vss_ne_1 = Vh - (Vo - self.IV(sep, Ri, Rh) + all_V[0])
                                
                # At this point, single source might heat one pt ionize other
                
                #self.IV(sep, Rh, Rc)
                
                
                
                
                #Vss_ne_1 = 4. * np.pi * (Rh - Ri)**3 / 3. \
                #         - 0.5 * all_V[1] - all_V[0]
                # Why the factor of 1/2? all_V[0] needed?
                         
                Vss_ne_2 = Vss_ne_1 

            elif term == 'im':
                Vo = all_V[0]
                
                delta_B = self._B(z, zeta)
                
                # Multiply by Mi just to do the integral in log
                _P1_integrand = dndm * Vo * (1. + delta_B)
                #P1 = np.trapz(_P1_integrand[iM:], x=np.log(Mi[iM:]))
                
                exp_int1 = np.exp(-np.trapz(_P1_integrand[iM:] * Mi[iM:],
                    x=np.log(Mi[iM:])))
                P1 = (1. - exp_int1)
                
                
                #Vss_ne_1 = Vss_ne_2 = np.zeros_like(Ri)
        
                bh = self.halos.Bias(z)
                bb = self.bubble_bias(z, zeta)
                xi = self.spline_cf_mm(z)(sep)
                
                Vi = 4. * np.pi * Ri**3 / 3.
                
                Vss_ne_1 = Vi - Vo
                Vss_ne_2 = Vss_ne_1
                
                
                #dndm_h = self.halos.tab_dndm[iz_hmf]
                #iM_h = np.argmin(np.abs(Mi - self.Mmin(z)))
                
                _prob_i_d = dndm * bb * xi_dd[i] * Vi * Mi
                prob_i_d = np.trapz(_prob_i_d[iM:], x=np.log(Mi[iM:]))
                
                #_Pout = bh * dndm_h
                #Pout = np.trapz(_Pout[iM_h:], x=np.log(self.halos.M[iM_h:]))
                #_Pout_int = 
                #Pout = np.trapz(_Pout_int[iM:], x=np.log(Mi[iM:]))
                Pout = prob_i_d
                
                #print(z, i, sep, Pin, Pout)
                
                B1[i] = P1
                #B2[i] = Pout
                
                continue
                                                                    
            elif term == 'cc':
                
                #if self.pf['powspec_lya_method'] > 0:
                #    Vo = self.IV(sep, Rc, Rc)
                #    Vss_ne_1 = 4. * np.pi * Rc**3 / 3. - Vo
                #    Vss_ne_2 = Vss_ne_1
                #else:
                Vo = all_V[-1]
                Vss_ne_1 = 4. * np.pi * (Rc - Rh)**3 / 3. \
                         - (self.IV(sep, Rc, Rc) - self.IV(sep, Rh, Rc))
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
        
            elif term == 'hc':
                Vo = all_V[-2]
        
                # One point in cold shell of one bubble, another in
                # heated shell of completely separate bubble.
                # Need to be a little careful here!
                Vss_ne_1 = 4. * np.pi * (Rh - Ri)**3 / 3. \
                         - self.IV(sep, Rh, Rc)
                Vss_ne_2 = 4. * np.pi * (Rc - Rh)**3 / 3. \
                         - (self.IV(sep, Rc, Rc) - self.IV(sep, Rh, Rc)) \
                         - self.IV(sep, Rh, Rc)
                                            
                limiter = None
        
            elif term == 'ih':
                #Vo_sh_r1, Vo_sh_r2, Vo_sh_r3 = \
                #    self.overlap_region_shell(sep, Ri, Rh)
                #Vo = 2. * Vo_sh_r2 - Vo_sh_r3
                Vo = all_V[1]
                                    
                Vss_ne_1 = 4. * np.pi * Ri**3 / 3. \
                         - self.IV(sep, Ri, Rh)
                Vss_ne_2 = 4. * np.pi * (Rh - Ri)**3 / 3. \
                         - self.IV(sep, Ri, Rh)
        
                limiter = None
                
            elif term == 'ic':
                Vo = all_V[2]
                Vss_ne_1 = 4. * np.pi * Ri**3 / 3. \
                         - (self.IV(sep, Ri, Rc) - self.IV(sep, Ri, Rh))
                Vss_ne_2 = 4. * np.pi * (Rc - Rh)**3 / 3. \
                         - (self.IV(sep, Ri, Rc) - self.IV(sep, Ri, Rh))
                
                limiter = None
            
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
        
        
            ##
            # Currently, only make it here for auto- terms
            ##
        
            # Compute the one-bubble term
            integrand1 = dndm * Vo
                     
            exp_int1 = np.exp(-np.trapz(integrand1[iM:] * Mi[iM:],
                x=np.log(Mi[iM:])))
            P1 = (1. - exp_int1)
            #P1 = np.trapz(integrand1[iM:] * Mi[iM:],
            #    x=np.log(Mi[iM:]))
            
            #print(sep, P1 / P1_2)
                                
            # Start chugging along on two-bubble term   
            
            #if np.any(Vss_ne_1 < 0):
            #    print('Vss_ne_1 ={} at R={}'.format(Vss_ne_1, sep))
            #else:
            #    print('Vss_ne_1 =OK at R={}'.format(sep))
            #if np.any(Vss_ne_2 < 0):     
            #    print('Vss_ne_2 < 0 at', sep)
                             
            integrand2_1 = dndm * np.maximum(Vss_ne_1, 0.0)
            integrand2_2 = dndm * np.maximum(Vss_ne_2, 0.0)
        
            exp_int2 = np.exp(-np.trapz(integrand2_1[iM:] * Mi[iM:], 
                x=np.log(Mi[iM:])))
            exp_int2_ex = np.exp(-np.trapz(integrand2_2[iM:] * Mi[iM:] \
                * corr[iM:], x=np.log(Mi[iM:])))
                
            # Is this leading factor double counting in some sense?    
            P2 = exp_int1 * (1. - exp_int2) * (1. - exp_int2_ex)            
            
            #P2 = np.trapz(integrand2_1[iM:] * Mi[iM:], x=np.log(Mi[iM:])) \
            #   * np.trapz(integrand2_2[iM:] * Mi[iM:] * corr[iM:], 
            #        x=np.log(Mi[iM:]))
            
            B1[i] = P1
            B2[i] = P2            
            
        # Experimentation don't freak out
        #B2 *= np.exp(-R / 150.)
      
        #ip = np.argmin(np.abs(R - 0.05))
        #B1[R <= 0.05] = B1[ip] * (R[R <= 0.05] / 0.05)**-0.05
      
        BT = B1 + B2
       
        self._cache_jp_[z][term] = R, BT, B1, B2
        
        return BT, B1, B2


    def CorrelationFunction(self, z, zeta=None, R=None, Q=None, term='ii', 
        rescale=False, Rh=0.0, Rc=0.0, Th=500., Tc=1., 
        assume_saturated=True, include_xcorr=False, include_ion=True,
        include_temp=False, include_lya=False, include_density=True,
        include_21cm=True):
        """
        Compute the correlation function of some general term.
        
        """
        
        if (Q is None) or rescale:
            Q = self.BubbleFillingFactor(z, zeta, rescale)
        
        if R is None:
            use_R_tab = True
            R = self.halos.tab_R
        else:
            use_R_tab = False    
            
        Tcmb = self.cosm.TCMB(z)
        Tgas = self.cosm.Tgas(z)   
        
        if type(Rh) == np.ndarray:
            
            Ri, Mi, dndm = self.BubbleSizeDistribution(z, zeta, rescale, Q)
            
            assert np.allclose(np.diff(Rh / Ri), 0.0), \
                "No support for absolute scaling of hot bubbles yet."
                        
            Qh = np.minimum(Q * (Rh[0] / Ri[0])**3, 1- Q)
        else:
            Qh = 0 
            
        ##
        # Check cache for match
        ##
        cached_result = self._cache_cf(z, term)
        
        if cached_result is not None:
            _R, _cf = cached_result
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
                cf = np.interp(R, self.halos.tab_R, self.halos.tab_cf_mm[iz])

        ##
        # Ionization correlation function
        ##
        elif term == 'ii':
            
            if not include_ion:
                cf = np.zeros_like(R)    
                self._cache_cf_[z][term] = R, cf
                return cf
            
            jp_ii, jp_ii_1, jp_ii_2 = \
                self.JointProbability(z, zeta, R=R, Q=Q, term='ii')
                
            # Add optional correction to ensure limiting behavior?        
            if self.pf['ps_volfix']:
                if Q < 0.5:
                    ev_ii = jp_ii_1 + jp_ii_2
                else:
                    ev_ii = (1. - Q) * jp_ii_1 + Q**2
            else:    
                ev_ii = jp_ii
                                    
            cf = ev_ii - Q**2
        
        ##
        # Ionization-density cross correlation function
        ##
        elif term == 'im':
            
            if (not include_xcorr) or (not include_ion) or (not include_density):
                cf = np.zeros_like(R)    
                self._cache_cf_[z][term] = R, cf
                return cf
            
            jp_ii, jp_ii_1, jp_ii_2 = \
                self.JointProbability(z, zeta, R, Q, term='ii')

            jp, jp_1h, jp_2h = \
                self.JointProbability(z, zeta, R, Q, term='im')
        
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
            
            if not include_temp:
                cf = np.zeros_like(R)    
                self._cache_cf_[z][term] = R, cf
                return cf
            
            jp_hh, jp_hh_1, jp_hh_2 = \
                self.JointProbability(z, zeta, R=R, Q=None, term='hh', Rh=Rh)
                
            Tref = Tgas
            delta_T = Tref / Th - 1.    
                
            #if ii <= 1:
            
            # Contrast of hot regions.
            C = (Tcmb / (Th - Tcmb)) * delta_T \
                 / (1. + delta_T)
            Csq = C**2
            
            # Do we want Qh here not including ionized regions?
            Cav = Qh * C
            
            print(z, Cav)
            
            ev_hh = Csq * jp_hh
            
            cf = ev_hh - Cav**2
            
            #else:
            #    # Remember, this is for the hot/cold term
            #    Csq = (Tcmb / (Tk - Tcmb)) * delta_T[0] * delta_T[1] \
            #        / (1. + delta_T[0]) / (1. + delta_T[1])
            #    C = np.sqrt(C)    
            
                
                
                
                
                
                
                
                
            # Add optional correction to ensure limiting behavior?        
            #if self.pf['ps_volfix']:
            #    if Q < 0.5:
            #        ev_ii = jp_ii_1 + jp_ii_2
            #    else:
            #        ev_ii = (1. - Q) * jp_ii_1 + Q**2
            #else:    
            #    ev_ii = jp_ii
                           
            #ev_hh = jp_hh               
            #                        
            #cf = ev_hh - Qh**2
            
        
        ##
        # 21-cm correlation function
        ##  
        elif term == '21':
            
            if not include_21cm:
                cf = np.zeros_like(R)    
                self._cache_cf_[z][term] = R, cf
                return cf
            
            # Ionization-ionization correlation function
            xi_ii = self.CorrelationFunction(z, zeta, R=R, Q=Q, term='ii', 
                rescale=rescale)
            
            # Correlation function in neutral fraction same by linearity.
            xi_nn = xi_ii
            
            # Matter correlation function    
            xi_mm = self.CorrelationFunction(z, zeta, R=R, term='mm', 
                rescale=rescale)
                
            if include_xcorr:
                #raise NotImplemented('help')
                xi_im = self.CorrelationFunction(z, zeta, R=R, Q=Q, term='im', 
                    rescale=rescale)
                xi_nm = -xi_im    
            else:
                xi_im = xi_nm = 0.0
            
            xbar = (1. - Q)
                        
            xi_21 = xi_nn * (1. + xi_mm) + xbar**2 * xi_mm + \
                xi_nm * (xi_nm + 2. * xbar)
            
            if not assume_saturated:
            
                # The two easiest terms in the unsaturated limit are those
                # not involving the density, <x x' C'> and <x x' C C'>.
                # Under the binary field(s) approach, we can just write
                # each of these terms down
                ev_xi_cop = data['Ch'] * data['jp_ih'] \
                          + data['Ch'] * data['jp_ic']
                
                ev_cd = data['ev_dco']
                ev_cc = data['ev_coco']
                ev_xx = 1. - 2. * xibar + data['ev_ii']
                ev_xc = data['avg_C'] - ev_xi_cop
                #ev_xxc = xbar * data['avg_C'] - ev_xi_cop
                #ev_xxcc = ev_xx * ev_cc + avg_xC**2 + ev_xc**2
                ev_xxc = 0.0
                ev_xxcc = ev_cc * (1. + xi_dd)
                ev_xxcd = ev_cd
                
                # <x x'>
                data['ev_xx'] = ev_xx
                
                # <x C'>
                data['ev_xc'] = ev_xc
                
                # <x x' C>
                data['ev_xxc'] = ev_xxc
                
                # <x x' C C'>
                data['ev_xxcc'] = ev_xxcc
                
                # <x x' C d'>
                data['ev_xxcd'] = ev_xxcd
                
                # Eq. 33 in write-up
                phi_u = 2. * ev_xxc + ev_xxcc + 2. * ev_xxcd
                                
                # Need to make sure this doesn't get saved at native resolution!
                #data['phi_u'] = phi_u
                    
                data['cf_21'] = data['cf_21_s'] + phi_u #\
                    #- 2. * xbar * avg_xC
                    
                Cbar = 1.#data['avg_C']
                data['cf_21'] = xi_CC * (1. + xi_dd) + Cbar**2 * xi_dd #+ \
                    #xi_Cd* (xi_Cd + 2. * Cbar)
                
                
            cf = xi_21
        
        else:
            raise NotImplemented('help')
        
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
    
    def CorrelationFunctionFromPS(self, R, ps, k=None, damp=None):
        return self.halos.InverseFT3D(R, ps, k, damp)
    def PowerSpectrumFromCF(self, k, cf, R=None, damp=None):
        
        #self._cache_ps_[z][term] = R, cf
        
        return self.halos.FT3D(k, cf, R, damp)    
        
