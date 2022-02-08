"""

FluctuatingBackground.py

Author: Jordan M_brocha
Affiliation: UCLA
Created on: Mon Oct 10 14:29:54 PDT 2016

Description:

"""

import numpy as np
from math import factorial
from ..physics import Cosmology
from ..util import ParameterFile
from ..util.Stats import bin_c2e
from scipy.special import erfinv
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from scipy.integrate import quad, simps
from ..physics.Hydrogen import Hydrogen
from ..physics.HaloModel import HaloModel
from ..populations.Composite import CompositePopulation
from ..physics.CrossSections import PhotoIonizationCrossSection
from ..physics.Constants import g_per_msun, cm_per_mpc, dnu, s_per_yr, c, \
    s_per_myr, erg_per_ev, k_B, m_p, dnu, g_per_msun

root2 = np.sqrt(2.)
four_pi = 4. * np.pi

class Fluctuations(object): # pragma: no cover
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

        self._done = {}

    @property
    def zeta(self):
        if not hasattr(self, '_zeta'):
            raise AttributeError('Must set zeta by hand!')
        return self._zeta

    @zeta.setter
    def zeta(self, value):
        self._zeta = value

    @property
    def zeta_X(self):
        if not hasattr(self, '_zeta_X'):
            raise AttributeError('Must set zeta_X by hand!')
        return self._zeta_X

    @zeta_X.setter
    def zeta_X(self, value):
        self._zeta_X = value

    @property
    def hydr(self):
        if not hasattr(self, '_hydr'):
            if self.grid is None:
                self._hydr = Hydrogen(**self.pf)
            else:
                self._hydr = self.grid.hydr

        return self._hydr

    @property
    def xset(self):
        if not hasattr(self, '_xset'):


            xset_pars = \
            {
             'xset_window': 'tophat-real',
             'xset_barrier': 'constant',
             'xset_pdf': 'gaussian',
            }

            xset = ares.physics.ExcursionSet(**xset_pars)
            xset.tab_M = pop.halos.tab_M
            xset.tab_sigma = pop.halos.tab_sigma
            xset.tab_ps = pop.halos.tab_ps_lin
            xset.tab_z = pop.halos.tab_z
            xset.tab_k = pop.halos.tab_k_lin
            xset.tab_growth = pop.halos.tab_growth

            self._xset = xset

        return self._xset

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

    def overlap_volumes(self, dr, R1, R2):
        """
        Overlap volumes, i.e., volumes in which a source affects two points
        in different ways. For example, V11 is the volume in which a source
        ionizes both points (at separation `dr`), V12 is the volume in which
        a source ionizes one point and heats the other, and so on.

        In this order: V11, V12, V13, V22, V23, V33
        """

        IV = self.IV
        V1 = 4. * np.pi * R1**3 / 3.

        if self.pf['ps_temp_model'] == 1:
            V2 = 4. * np.pi * (R2**3 - R1**3) / 3.
        else:
            V2 = 4. * np.pi * R2**3 / 3.

        Vt = 4. * np.pi * R2**3 / 3.

        V11 = IV(dr, R1, R1)

        if self.pf['ps_include_temp'] and self.pf['ps_temp_model'] == 2:
            V12 = V1
        else:
            V12 = 2 * IV(dr, R1, R2) - IV(dr, R1, R1)

        V22 = IV(dr, R2, R2)
        if self.pf['ps_temp_model'] == 1:
            V22 += -2. * IV(dr, R1, R2) + IV(dr, R1, R1)

        if self.pf['ps_include_temp'] and self.pf['ps_temp_model'] == 1:
            V1n = V1 - IV(dr, R1, R2)
        elif self.pf['ps_include_temp'] and self.pf['ps_temp_model'] == 2:
            V1n = V1
        else:
            V1n = V1 - V11

        V2n = V2 - IV(dr, R2, R2)
        if self.pf['ps_temp_model'] == 1:
            V2n += IV(dr, R1, R2)

        # 'anything' to one point, 'nothing' to other.
        # Without temperature fluctuations, same as V1n
        if self.pf['ps_include_temp']:
            Van = Vt - IV(dr, R2, R2)
        else:
            Van = V1n

        return V11, V12, V22, V1n, V2n, Van

    def exclusion_volumes(self, dr, R1, R2, R3):
        """
        Volume in which a single source only affects one
        """
        pass

    @property
    def heating_ongoing(self):
        if not hasattr(self, '_heating_ongoing'):
            self._heating_ongoing = True
        return self._heating_ongoing

    @heating_ongoing.setter
    def heating_ongoing(self, value):
        self._heating_ongoing = value

    def BubbleShellFillingFactor(self, z, R_s=None):
        """

        """

        # Hard exit.
        if not self.pf['ps_include_temp']:
            return 0.0

        Qi = self.MeanIonizedFraction(z)

        if self.pf['ps_temp_model'] == 1:
            R_i, M_b, dndm_b = self.BubbleSizeDistribution(z)


            if Qi == 1:
                return 0.0

            if type(R_s) is np.ndarray:
                nz = R_i > 0

                const_rsize = np.allclose(np.diff(R_s[nz==1] / R_i[nz==1]), 0.0)

                if const_rsize:
                    fvol = (R_s[0] / R_i[0])**3 - 1.

                    Qh = Qi * fvol

                else:

                    V = 4. * np.pi * (R_s**3 - R_i**3) / 3.

                    Mmin = self.Mmin(z) * self.zeta

                    Qh = self.get_prob(z, M_b, dndm_b, Mmin, V,
                        exp=False, ep=0.0, Mmax=None)

                    #raise NotImplemented("No support for absolute scaling of hot bubbles yet.")

                if (Qh > (1. - Qi) * 1.): #or Qh > 0.5: #or Qi > 0.5:
                    self.heating_ongoing = 0

                Qh = np.minimum(Qh, 1. - Qi)

                return Qh

            else:
                # This will get called if temperature fluctuations are off
                return 0.0

        elif self.pf['ps_temp_model'] == 2:
            Mmin = self.Mmin(z) * self.zeta_X
            R_i, M_b, dndm_b = self.BubbleSizeDistribution(z, ion=False)
            V = 4. * np.pi * R_i**3 / 3.
            Qh = self.get_prob(z, M_b, dndm_b, Mmin, V,
                exp=False, ep=0.0, Mmax=None)


            #Qh = self.BubbleFillingFactor(z, ion=False)
            #print('Qh', Qh)
            return np.minimum(Qh, 1. - Qi)
        else:
            raise NotImplemented('Uncrecognized option for BSD.')

        #return min(Qh, 1.), min(Qc, 1.)

    @property
    def bsd_model(self):
        if self.pf['bubble_size_dist'] is None:
            return None
        else:
            return self.pf['bubble_size_dist'].lower()

    def MeanIonizedFraction(self, z, ion=True):
        Mmin = self.Mmin(z)
        logM = np.log10(Mmin)

        if ion:
            if not self.pf['ps_include_ion']:
                return 0.0

            zeta = self.zeta

            return np.minimum(1.0, zeta * self.halos.fcoll_2d(z, logM))
        else:
            if not self.pf['ps_include_temp']:
                return 0.0
            zeta = self.zeta_X

            # Assume that each heated region contains the same volume
            # of fully-ionized material.
            Qi = self.MeanIonizedFraction(z, ion=True)

            Qh = zeta * self.halos.fcoll_2d(z, logM) - Qi

            return np.minimum(1.0 - Qi, Qh)

    def delta_shell(self, z):
        """
        Relative density != relative over-density.
        """

        if not self.pf['ps_include_temp']:
            return 0.0

        if self.pf['ps_temp_model'] == 2:
            return self.delta_bubble_vol_weighted(z, ion=False)

        delta_i_bar = self.delta_bubble_vol_weighted(z)

        rdens = self.pf["bubble_shell_rdens_zone_0"]

        return rdens * (1. + delta_i_bar) - 1.

    def BulkDensity(self, z, R_s):
        Qi = self.MeanIonizedFraction(z)
        #Qh = self.BubbleShellFillingFactor(z, R_s)
        Qh = self.MeanIonizedFraction(z, ion=False)

        delta_i_bar = self.delta_bubble_vol_weighted(z)
        delta_h_bar = self.delta_shell(z)

        if self.pf['ps_igm_model'] == 2:
            delta_hal_bar = self.mean_halo_overdensity(z)
            Qhal = self.Qhal(z, Mmax=self.Mmin(z))
        else:
            Qhal = 0.0
            delta_hal_bar = 0.0

        return -(delta_i_bar * Qi + delta_h_bar * Qh + delta_hal_bar * Qhal) \
            / (1. - Qi - Qh - Qhal)

    def BubbleFillingFactor(self, z, ion=True, rescale=True):
        """
        Fraction of volume filled by bubbles.

        This is never actually used, but for reference, the mean ionized
        fraction would be 1 - exp(-this). What we actually do is re-normalize
        the bubble size distribution to guarantee Q = zeta * fcoll. See
        MeanIonizedFraction and BubbleSizeDistribution for more details.
        """

        if self.bsd_model is not None:
            if ion:
                zeta = self.zeta
            else:
                zeta = self.zeta_X

        if self.bsd_model is None:
            R_i = self.pf['bubble_size']
            V_i = 4. * np.pi * R_i**3 / 3.
            ni = self.BubbleDensity(z)

            Qi = 1. - np.exp(-ni * V_i)

        elif self.bsd_model in ['fzh04', 'hmf']:

            # Smallest bubble is one around smallest halo.
            # Don't actually need its mass, just need index to correctly
            # truncate integral.
            Mmin = self.Mmin(z) * zeta

            # M_b should just be self.m? No.
            R_i, M_b, dndm_b = self.BubbleSizeDistribution(z, ion=ion,
                rescale=rescale)
            V_i = 4. * np.pi * R_i**3 / 3.

            iM = np.argmin(np.abs(Mmin - M_b))

            _Qi = np.trapz(dndm_b[iM:] * M_b[iM:] * V_i[iM:],
                x=np.log(M_b[iM:]))
            Qi = 1. - np.exp(-_Qi)

            # This means reionization is over.
            if self.bsd_model == 'fzh04':
                if self._B0(z, zeta) <= 0:
                    return 1.

        else:
            raise NotImplemented('Uncrecognized option for BSD.')

        return min(Qi, 1.)

        # Grab heated phase to enforce BC
        #Rs = self.BubbleShellRadius(z, R_i)
        #Vsh = 4. * np.pi * (Rs - R_i)**3 / 3.
        #Qh = np.trapz(dndm * Vsh * M_b, x=np.log(M_b))

        #if lya and self.pf['bubble_pod_size_func'] in [None, 'const', 'linear']:
        #    Rc = self.BubblePodRadius(z, R_i, zeta, zeta_lya)
        #    Vc = 4. * np.pi * (Rc - R_i)**3 / 3.
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

    def tab_bubble_bias(self, zeta):
        if not hasattr(self, '_tab_bubble_bias'):
            func = lambda z: self._fzh04_eq22(z, zeta)
            self._tab_bubble_bias = np.array(map(func, self.halos.tab_z_ps))

        return self._tab_bubble_bias

    def _fzh04_eq22(self, z, ion=True):

        if ion:
            zeta = self.zeta
        else:
            zeta = self.zeta_X


        iz = np.argmin(np.abs(z - self.halos.tab_z))
        s = self.sigma
        S = s**2

        #return 1. + ((self.LinearBarrier(z, zeta, zeta) / S - (1. / self._B0(z, zeta))) \
        #    / self._growth_factor(z))

        return 1. + (self._B0(z, zeta)**2 / S / self._B(z, zeta, zeta))

    def bubble_bias(self, z, ion=True):
        """
        Eq. 9.24 in Loeb & Furlanetto (2013) or Eq. 22 in FZH04.
        """

        return self._fzh04_eq22(z, ion)

        #iz = np.argmin(np.abs(z - self.halos.tab_z_ps))
        #
        #x, y = self.halos.tab_z_ps, self.tab_bubble_bias(zeta)[iz]
        #
        #
        #
        #m = (y[-1] - y[-2]) / (x[-1] - x[-2])
        #
        #return m * z + y[-1]

        #iz = np.argmin(np.abs(z - self.halos.tab_z))
        #s = self.sigma
        #S = s**2
        #
        ##return 1. + ((self.LinearBarrier(z, zeta, zeta) / S - (1. / self._B0(z, zeta))) \
        ##    / self._growth_factor(z))
        #
        #fzh04 = 1. + (self._B0(z, zeta)**2 / S / self._B(z, zeta, zeta))
        #
        #return fzh04

    def mean_bubble_bias(self, z, ion=True):
        """
        """

        R, M_b, dndm_b = self.BubbleSizeDistribution(z, ion=ion)

        #if ('h' in term) or ('c' in term) and self.pf['powspec_temp_method'] == 'shell':
        #    R_s, Rc = self.BubbleShellRadius(z, R_i)
        #    R = R_s
        #else:

        if ion:
            zeta = self.zeta
        else:
            zeta = self.zeta_X

        V = 4. * np.pi * R**3 / 3.

        Mmin = self.Mmin(z) * zeta
        iM = np.argmin(np.abs(Mmin - self.m))
        bHII = self.bubble_bias(z, ion)

        #tmp = dndm[iM:]
        #print(z, len(tmp[np.isnan(tmp)]), len(bHII[np.isnan(bHII)]))

        #imax = int(min(np.argwhere(np.isnan(R_i))))

        if ion and self.pf['ps_include_ion']:
            Qi = self.MeanIonizedFraction(z)
        elif ion and not self.pf['ps_include_ion']:
            raise NotImplemented('help')
        elif (not ion) and self.pf['ps_include_temp']:
            Qi = self.MeanIonizedFraction(z, ion=False)
        elif ion and self.pf['ps_include_temp']:
            Qi = self.MeanIonizedFraction(z, ion=False)
        else:
            raise NotImplemented('help')

        return np.trapz(dndm_b[iM:] * V[iM:] * bHII[iM:] * M_b[iM:],
            x=np.log(M_b[iM:])) / Qi

    #def delta_bubble_mass_weighted(self, z, zeta):
    #    if self._B0(z, zeta) <= 0:
    #        return 0.
    #
    #    R_i, M_b, dndm_b = self.BubbleSizeDistribution(z, zeta)
    #    V_i = 4. * np.pi * R_i**3 / 3.
    #
    #    Mmin = self.Mmin(z) * zeta
    #    iM = np.argmin(np.abs(Mmin - self.m))
    #    B = self._B(z, zeta)
    #    rho0 = self.cosm.mean_density0
    #
    #    dm_ddel = rho0 * V_i
    #
    #    return simps(B[iM:] * dndm_b[iM:] * M_b[iM:], x=np.log(M_b[iM:]))

    def delta_bubble_vol_weighted(self, z, ion=True):
        if not self.pf['ps_include_ion']:
            return 0.0

        if not self.pf['ps_include_xcorr_ion_rho']:
            return 0.0

        if ion:
            zeta = self.zeta
        else:
            zeta = self.zeta_X

        if self._B0(z, zeta) <= 0:
            return 0.

        R_i, M_b, dndm_b = self.BubbleSizeDistribution(z, ion=ion)
        V_i = 4. * np.pi * R_i**3 / 3.

        Mmin = self.Mmin(z) * zeta
        iM = np.argmin(np.abs(Mmin - self.m))
        B = self._B(z, ion=ion)

        return np.trapz(B[iM:] * dndm_b[iM:] * V_i[iM:] * M_b[iM:],
            x=np.log(M_b[iM:]))

   #def mean_bubble_overdensity(self, z, zeta):
   #    if self._B0(z, zeta) <= 0:
   #        return 0.
   #
   #    R_i, M_b, dndm_b = self.BubbleSizeDistribution(z, zeta)
   #    V_i = 4. * np.pi * R_i**3 / 3.
   #
   #    Mmin = self.Mmin(z) * zeta
   #    iM = np.argmin(np.abs(Mmin - self.m))
   #    B = self._B(z, zeta)
   #    rho0 = self.cosm.mean_density0
   #
   #    dm_ddel = rho0 * V_i
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

        return np.trapz(M_h * dndm_h, x=np.log(M_h))

    def spline_cf_mm(self, z):
        if not hasattr(self, '_spline_cf_mm_'):
            self._spline_cf_mm_ = {}

        if z not in self._spline_cf_mm_:
            iz = np.argmin(np.abs(z - self.halos.tab_z_ps))

            self._spline_cf_mm_[z] = interp1d(np.log(self.halos.tab_R),
                self.halos.tab_cf_mm[iz], kind='cubic', bounds_error=False,
                fill_value=0.0)

        return self._spline_cf_mm_[z]

    def excess_probability(self, z, R, ion=True):
        """
        This is the excess probability that a point is ionized given that
        we already know another point (at distance r) is ionized.
        """

        # Function of bubble mass (bubble size)
        bHII = self.bubble_bias(z, ion)
        bbar = self.mean_bubble_bias(z, ion)

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

    def _B0(self, z, ion=True):

        if ion:
            zeta = self.zeta
        else:
            zeta = self.zeta_X

        iz = np.argmin(np.abs(z - self.halos.tab_z))
        s = self.sigma

        # Variance on scale of smallest collapsed object
        sigma_min = self.sigma_min(z)

        return self._delta_c(z) - root2 * self._K(zeta) * sigma_min

    def _B1(self, z, ion=True):
        if ion:
            zeta = self.zeta
        else:
            zeta = self.zeta_X

        iz = np.argmin(np.abs(z - self.halos.tab_z))
        s = self.sigma #* self.halos.growth_factor[iz]

        sigma_min = self.sigma_min(z)

        return self._K(zeta) / np.sqrt(2. * sigma_min**2)

    def _B(self, z, ion=True, zeta_min=None):
        return self.LinearBarrier(z, ion, zeta_min=zeta_min)

    def LinearBarrier(self, z, ion=True, zeta_min=None):

        if ion:
            zeta = self.zeta
        else:
            zeta = self.zeta_X

        iz = np.argmin(np.abs(z - self.halos.tab_z))
        s = self.sigma #/ self.halos.growth_factor[iz]

        if zeta_min is None:
            zeta_min = zeta

        return self._B0(z, ion) + self._B1(z, ion) * s**2

    def Barrier(self, z, ion=True, zeta_min=None):
        """
        Full barrier.
        """

        if ion:
            zeta = self.zeta
        else:
            zeta = self.zeta_X

        if zeta_min is None:
            zeta_min = zeta

        #iz = np.argmin(np.abs(z - self.halos.tab_z))
        #D = self.halos.growth_factor[iz]

        sigma_min = self.sigma_min(z)
        #Mmin = self.Mmin(z)
        #sigma_min = np.interp(Mmin, self.halos.M, self.halos.sigma_0)

        delta = self._delta_c(z)

        return delta - np.sqrt(2.) * self._K(zeta) \
            * np.sqrt(sigma_min**2 - self.sigma**2)

        #return self.cosm.delta_c0 - np.sqrt(2.) * self._K(zeta) \
        #    * np.sqrt(sigma_min**2 - s**2)

    def sigma_min(self, z):
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
            self._m = 10**np.arange(5, 18.1, 0.1)
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

    def BubbleSizeDistribution(self, z, ion=True, rescale=True):
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

        if ion:
            zeta = self.zeta
        else:
            zeta = self.zeta_X

        if ion and not self.pf['ps_include_ion']:
            R_i = M_b = dndm = np.zeros_like(self.m)
            return R_i, M_b, dndm
        if (not ion) and not self.pf['ps_include_temp']:
            R_i = M_b = dndm = np.zeros_like(self.m)
            return R_i, M_b, dndm

        reionization_over = False

        # Comoving matter density
        rho0_m = self.cosm.mean_density0
        rho0_b = rho0_m * self.cosm.fbaryon

        # Mean (over-)density of bubble material
        delta_B = self._B(z, ion)

        if self.bsd_model is None:
            if self.pf['bubble_density'] is not None:
                R_i = self.pf['bubble_size']
                M_b = (4. * np.pi * Rb**3 / 3.) * rho0_m
                dndm = self.pf['bubble_density']
            else:
                raise NotImplementedError('help')

        elif self.bsd_model == 'hmf':
            M_b = self.halos.tab_M * zeta
            # Assumes bubble material is at cosmic mean density
            R_i = (3. * M_b / rho0_b / 4. / np.pi)**(1./3.)
            iz = np.argmin(np.abs(z - self.halos.tab_z))
            dndm = self.halos.tab_dndm[iz].copy()

        elif self.bsd_model == 'fzh04':
            # Just use array of halo mass as array of ionized region masses.
            # Arbitrary at this point, just need an array of masses.
            # Plus, this way, the sigma's from the HMF are OK.
            M_b = self.m

            # Radius of ionized regions as function of delta (mass)
            R_i = (3. * M_b / rho0_m / (1. + delta_B) / 4. / np.pi)**(1./3.)

            V_i = four_pi * R_i**3 / 3.

            # This is Eq. 9.38 from Steve's book.
            # The factors of 2, S, and M_b are from using dlns instead of
            # dS (where S=s^2)
            dndm = rho0_m * self.pcross(z, ion) * 2 * np.abs(self.dlns_dlnm) \
                * self.sigma**2 / M_b**2

            # Reionization is over!
            # Only use barrier condition if we haven't asked to rescale
            # or supplied Q ourselves.
            if self._B0(z, ion) <= 0:
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
            Qi = np.trapz(dndm[iM:] * V_i[iM:] * M_b[iM:], x=np.log(M_b[iM:]))
            xibar = self.MeanIonizedFraction(z, ion=ion)
            dndm *= -np.log(1. - xibar) / Qi

        return R_i, M_b, dndm

    def pcross(self, z, ion=True):
        """
        Up-crossing probability.
        """

        if ion:
            zeta = self.zeta
        else:
            zeta = self.zeta_X

        S = self.sigma**2
        Mmin = self.Mmin(z) #* zeta # doesn't matter for zeta=const
        if type(zeta) == np.ndarray:
            raise NotImplemented('this is wrong.')
            zeta_min = np.interp(Mmin, self.m, zeta)
        else:
            zeta_min = zeta

        zeros = np.zeros_like(self.sigma)

        B0 = self._B0(z, ion)
        B1 = self._B1(z, ion)
        Bl = self.LinearBarrier(z, ion=ion, zeta_min=zeta_min)
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

    def get_Nion(self, z, R_i):
        return 4. * np.pi * (R_i * cm_per_mpc / (1. + z))**3 \
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
            #print("Loaded cf_{} at z={} from cache.".format(term, z))
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
            self._is_Rs_const = True
        return self._is_Rs_const

    @is_Rs_const.setter
    def is_Rs_const(self, value):
        self._is_Rs_const = value

    def _cache_Vo(self, z):
        if not hasattr(self, '_cache_Vo_'):
            self._cache_Vo_ = {}

        if z in self._cache_Vo_:
            return self._cache_Vo_[z]

        #if self.is_Rs_const and len(self._cache_Vo_.keys()) > 0:
        #    return self._cache_Vo_[self._cache_Vo_.keys()[0]]

        return None

    def _cache_IV(self, z):
        if not hasattr(self, '_cache_IV_'):
            self._cache_IV_ = {}

        if z in self._cache_IV_:
            return self._cache_IV_[z]

        #if self.is_Rs_const and len(self._cache_IV_.keys()) > 0:
        #    return self._cache_IV_[self._cache_IV_.keys()[0]]

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

    def Qhal(self, z, Mmin=None, Mmax=None):
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

        if Mmin is not None:
            imin = np.argmin(np.abs(M_h - Mmin))
        else:
            imin = 0

        if Mmax is not None:
            imax = np.argmin(np.abs(M_h - Mmax))
        else:
            imax = None

        integ = dndm_h * Vvir * M_h

        Q_hal = 1. - np.exp(-np.trapz(integ[imin:imax],
            x=np.log(M_h[imin:imax])))

        return Q_hal

        #return self.get_prob(z, M_h, dndm_h, Mmin, Vvir, exp=False, ep=0.0,
        #    Mmax=Mmax)

    def ExpectationValue1pt(self, z, term='i', R_s=None, R3=None,
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

        Qi = self.MeanIonizedFraction(z)
        Qh = self.MeanIonizedFraction(z, ion=False)

        if self.pf['ps_igm_model'] == 2:
            Qhal = self.Qhal(z, Mmax=self.Mmin(z))
            del_hal = self.mean_halo_overdensity(z)
        else:
            Qhal = 0.0
            del_hal = 0.0

        Qb = 1. - Qi - Qh - Qhal
        Tcmb = self.cosm.TCMB(z)
        del_i = self.delta_bubble_vol_weighted(z)
        del_h = self.delta_shell(z)

        del_b = self.BulkDensity(z, R_s)
        ch = self.TempToContrast(z, Th=Th, Tk=Tk, Ts=Ts, Ja=Ja)

        if Ts is not None:
            cb = Tcmb / Ts
        else:
            cb = 0.0

        ##
        # Otherwise, get to it.
        ##
        if term == 'b':
            val = 1. - Qi - Qh - Qhal
        elif term == 'i':
            val = Qi
        elif term == 'n':
            val = 1. - Qi
        elif term == 'h':
            assert R_s is not None
            val = Qh
        elif term in ['m', 'd']:
            val = 0.0
        elif term in ['n*d', 'i*d']:
            # <xd> = <(1-x_i)d> = <d> - <x_i d> = - <x_i d>
            if self.pf['ps_include_xcorr_ion_rho']:

                if term == 'i*d':
                    val = Qi * del_i
                else:
                    val = -Qi * del_i
            else:
                val = 0.0
        elif term == 'pc':
            # <psi * c> = <x (1 + d) c>
            # = <(1 - i) (1 + d) c> = <(1 + d) c> - <i (1 + d) c>
            # ...
            # = <c> + <cd>
            avg_c = Qh * ch + Qb * cb
            if self.pf['ps_include_xcorr_hot_rho']:
                val = avg_c \
                    + Qh * ch * del_h \
                    + Qb * cb * del_b
            else:
                val = avg_c
        elif term in ['ppc', 'ppcc']:
            avg_psi = self.ExpectationValue1pt(z, term='psi',
                R_s=R_s, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)
            avg_c = self.ExpectationValue1pt(z, term='c',
                R_s=R_s, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)

            if term == 'ppc':
                val = avg_psi**2 * avg_c
            else:
                val = avg_psi**2 * avg_c**2

            #cc = Qh**2 * ch**2 \
            #   + 2 * Qh * Qb * ch * cb \
            #   + Qb**2 * cb**2
            #ccd = Qh**2 * ch**2 * delta_h_bar \
            #   + Qh * Qb * ch * cb * delta_h_bar \
            #   + Qh * Qb * ch * cb * delta_b_bar \
            #   + Qb**2 * cb**2 * delta_b_bar

        elif term == 'c*d':
            if self.pf['ps_include_xcorr_hot_rho']:
                val = Qh * ch * del_h + Qb * cb * del_b
            else:
                val = 0.0
        elif term.strip() == 'i*h':
            val = 0.0
        elif term.strip() == 'n*h':
            # <xh> = <h> - <x_i h> = <h>
            val = Qh
        elif term.strip() == 'i*c':
            val = 0.0
        elif term == 'c':
            val = ch * Qh + cb * Qb
        elif term.strip() == 'n*c':
            # <xc> = <c> - <x_i c>
            val = ch * Qh
        elif term == 'psi':
            # <psi> = <x (1 + d)> = <x> + <xd> = 1 - <x_i> + <d> - <x_i d>
            #       = 1 - <x_i> - <x_i d>

            #avg_xd = self.ExpectationValue1pt(z, zeta, term='n*d',
            #    R_s=R_s, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)
            #avg_x = self.ExpectationValue1pt(z, zeta, term='n',
            #    R_s=R_s, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)
            #
            #val = avg_x + avg_xd

            avg_id = self.ExpectationValue1pt(z, term='i*d',
                R_s=R_s, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)
            avg_i = self.ExpectationValue1pt(z, term='i',
                R_s=R_s, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)

            val = 1. - avg_i - avg_id

        elif term == 'phi':
            # <phi> = <psi * (1 - c)> = <psi> - <psi * c>
            # <psi * c> = <x * c> + <x * c * d>
            #           = <c> - <x_i c> + <cd> - <x_i c * d>
            avg_psi = self.ExpectationValue1pt(z, term='psi',
                R_s=R_s, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)
            avg_psi_c = self.ExpectationValue1pt(z, term='pc',
                R_s=R_s, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)

            val = avg_psi - avg_psi_c

            # <phi> = <psi * (1 + c)> = <psi> + <psi * c>
            #
            # <psi * c> = <x * c> + <x * c * d>
            #           = <c> - <x_i c> + <cd> - <x_i c * d>

            #avg_xcd = self.ExpectationValue1pt(z, zeta, term='n*d*c',
            #    R_s=R_s, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)
            #
            ## Equivalent to <c> in binary field model.
            #ch = self.TempToContrast(z, Th=Th, Tk=Tk, Ts=Ts, Ja=Ja)
            #ci = self.BubbleContrast(z, Th=Th, Tk=Tk, Ts=Ts, Ja=Ja)
            #avg_c = self.ExpectationValue1pt(z, zeta, term='c',
            #    R_s=R_s, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)
            #avg_cd = self.ExpectationValue1pt(z, zeta, term='c*d',
            #    R_s=R_s, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)
            #avg_id = self.ExpectationValue1pt(z, zeta, term='i*d',
            #    R_s=R_s, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)
            #
            #avg_psi = self.ExpectationValue1pt(z, zeta, term='psi',
            #    R_s=R_s, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)
            #avg_psi_c = avg_c - ci * Qi + avg_cd - avg_id * ci
            #
            ## Tagged on these last two terms if c=1 (ionized regions)
            #val = avg_psi + avg_psi_c

        elif term == '21':
            # dTb = T0 * (1 + d21) = T0 * xHI * (1 + d) = T0 * psi
            # so  d21 = psi - 1
            if self.pf['ps_include_temp']:
                avg_phi = self.ExpectationValue1pt(z, term='phi',
                    R_s=R_s, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)
                val = avg_phi - 1.
            else:
                avg_psi = self.ExpectationValue1pt(z, term='psi',
                    R_s=R_s, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)
                val = avg_psi - 1.
        elif term == 'o':
            # <omega>^2 = <psi * c>^2 - 2 <psi> <psi * c>
            avg_psi = self.ExpectationValue1pt(z, term='psi',
                R_s=R_s, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)
            avg_psi_c = self.ExpectationValue1pt(z, term='pc',
                R_s=R_s, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)
            # <omega>^2 = <psi c>^2 - 2 <psi c> <psi>
            val = np.sqrt(avg_psi_c**2 - 2. * avg_psi_c * avg_psi)
        elif term == 'oo':
            avg_psi = self.ExpectationValue1pt(z, term='psi',
                R_s=R_s, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)
            avg_psi_c = self.ExpectationValue1pt(z, term='pc',
                R_s=R_s, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)
            # <omega>^2 = <psi c>^2 - 2 <psi c> <psi>
            val = avg_psi_c**2 - 2. * avg_psi_c * avg_psi

        else:
            raise ValueError('Don\' know how to handle <{}>'.format(term))

        self._cache_p_[z][term] = val

        return val

    @property
    def _getting_basics(self):
        if not hasattr(self, '_getting_basics_'):
            self._getting_basics_ = False
        return self._getting_basics_

    def get_basics(self, z, R, R_s, Th, Ts, Tk, Ja):

        self._getting_basics_ = True

        basics = {}
        for term in ['ii', 'ih', 'ib', 'hh', 'hb', 'bb']:
            cache = self._cache_jp(z, term)


            if self.pf['ps_include_temp'] and self.pf['ps_temp_model'] == 2:
                Qi = self.MeanIonizedFraction(z)
                Qh = self.MeanIonizedFraction(z, ion=False)

                if term == 'ih':
                    P = Qi * Qh * np.ones_like(R)
                    P1 = P2 = np.zeros_like(R)
                    basics[term] = P, P1, P2
                    continue
                elif term == 'ib':
                    P = Qi * (1. - Qi - Qh) * np.ones_like(R)
                    P1 = P2 = np.zeros_like(R)
                    basics[term] = P, P1, P2
                    continue
                elif term == 'hb':
                    P = Qh * (1. - Qi - Qh) * np.ones_like(R)
                    P1 = P2 = np.zeros_like(R)
                    basics[term] = P, P1, P2
                    continue
                #elif term == 'bb':
                #    P = (1. - Qi - Qh)**2 * np.ones_like(R)
                #    P1 = P2 = np.zeros_like(R)
                #    basics[term] = P, P1, P2
                #    continue

            if cache is None and term != 'bb':
                P, P1, P2 = self.ExpectationValue2pt(z,
                    R=R, term=term, R_s=R_s, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)
            elif cache is None and term == 'bb':
                P = 1. - (basics['ii'][0] + 2 * basics['ib'][0]
                  + 2 * basics['ih'][0] + basics['hh'][0] + 2 * basics['hb'][0])
                P1 = P2 = np.zeros_like(P)
                self._cache_jp_[z][term] = R, P, np.zeros_like(P), np.zeros_like(P)
            else:
                P, P1, P2 = cache[1:]

            basics[term] = P, P1, P2

        self._getting_basics_ = False

        return basics

    def ExpectationValue2pt(self, z, R, term='ii', R_s=None, R3=None,
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
        #cached_result = self._cache_jp(z, term)
        #
        #if cached_result is not None:
        #    _R, _jp, _jp1, _jp2 = cached_result
        #
        #    if _R.size == R.size:
        #        if np.allclose(_R, R):
        #            return cached_result[1:]
        #
        #    print("interpolating jp_{}".format(ii))
        #    return np.interp(R, _R, _jp), np.interp(R, _R, _jp1), np.interp(R, _R, _jp2)

        # Remember, we scaled the BSD so that these two things are equal
        # by construction.
        xibar = Q = Qi = self.MeanIonizedFraction(z)

        # Call this early so that heating_ongoing is set before anything
        # else can happen.
        #Qh = self.BubbleShellFillingFactor(z, R_s=R_s)
        Qh = self.MeanIonizedFraction(z, ion=False)

        delta_i_bar = self.delta_bubble_vol_weighted(z)
        delta_h_bar = self.delta_shell(z)
        delta_b_bar = self.BulkDensity(z, R_s)

        Tcmb = self.cosm.TCMB(z)
        ch = self.TempToContrast(z, Th=Th, Tk=Tk, Ts=Ts, Ja=Ja)
        if Ts is None:
            cb = 0.0
        else:
            cb = Tcmb / Ts

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

        # Some stuff we need
        R_i, M_b, dndm_b = self.BubbleSizeDistribution(z)
        V_i = 4. * np.pi * R_i**3 / 3.

        if self.pf['ps_include_temp']:
            if self.pf['ps_temp_model'] == 1:
                V_h = 4. * np.pi * (R_s**3 - R_i**3) / 3.
                V_ioh = 4. * np.pi * R_s**3 / 3.
                dndm_s = dndm_b
                M_s = M_b
                zeta_X = 0.0
            elif self.pf['ps_temp_model'] == 2:
                zeta_X = self.zeta_X
                R_s, M_s, dndm_s = self.BubbleSizeDistribution(z, ion=False)
                V_h = 4. * np.pi * R_s**3 / 3.
                V_ioh = 4. * np.pi * R_s**3 / 3.
            else:
                raise NotImplemented('help')
        else:
            zeta_X = 0

        if R_s is None:
            R_s = np.zeros_like(R_i)
            V_h = np.zeros_like(R_i)
            V_ioh = V_i
            zeta_X = 0.0

        if R3 is None:
            R3 = np.zeros_like(R_i)

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
            #elif 'i' in term:
            #    #self._cache_jp_[z][term] = R, Rzeros, Rzeros, Rzeros
            #    return Rzeros, Rzeros, Rzeros
            # also iid, iidd

        if not self.pf['ps_include_temp']:
            if ('c' in term) or ('h' in term):
                return Rzeros, Rzeros, Rzeros


        ##
        # Handy
        ##
        if not self._getting_basics:
            basics = self.get_basics(z, R, R_s=R_s, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)

            if term in basics:
                return basics[term]

            _P_ii, _P_ii_1, _P_ii_2 = basics['ii']
            _P_hh, _P_hh_1, _P_hh_2 = basics['hh']
            _P_bb, _P_bb_1, _P_bb_2 = basics['bb']
            _P_ih, _P_ih_1, _P_ih_2 = basics['ih']
            _P_ib, _P_ib_1, _P_ib_2 = basics['ib']
            _P_hb, _P_hb_1, _P_hb_2 = basics['hb']


        ##
        # Check for derived quantities like psi, phi
        ##
        if term == 'psi':
            # <psi psi'> = <x (1 + d) x' (1 + d')> = <xx'(1+d)(1+d')>
            #            = <xx'(1 + d + d' + dd')>
            #            = <xx'> + 2<xx'd> + <xx'dd'>

            #xx, xx1, xx2 = self.ExpectationValue2pt(z, zeta, R=R, term='nn',
            #    R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)
            #
            #xxd, xxd1, xxd2 = self.ExpectationValue2pt(z, zeta, R=R, term='nnd',
            #    R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)
            #
            #xxdd, xxdd1, xxdd2 = self.ExpectationValue2pt(z, zeta, R=R,
            #    term='xxdd', R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)
            #
            #ev2pt = xx + 2. * xxd + xxdd
            #
            ev2pt_1 = Rzeros
            ev2pt_2 = Rzeros

            # All in terms of ionized fraction perturbation.
            dd, dd1, dd2 = self.ExpectationValue2pt(z, R=R, term='dd',
                R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)
            ii, ii1, ii2 = self.ExpectationValue2pt(z, R=R, term='ii',
                R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)
            di, di1, di2 = self.ExpectationValue2pt(z, R=R, term='id',
                R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)
            iidd, on, tw = self.ExpectationValue2pt(z, R=R, term='iidd',
                R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)
            idd, on, tw = self.ExpectationValue2pt(z, R=R, term='idd',
                R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)
            iid, on, tw = self.ExpectationValue2pt(z, R=R, term='iid',
                R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)

            ev_id_1 = self.ExpectationValue1pt(z, term='i*d',
                R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)

            ev2pt = dd + ii - 2. * di + iidd - 2. * (idd - iid) \
                  + 1. - 2 * Qi - 2 * ev_id_1

            #self._cache_jp_[z][term] = R, ev2pt, ev2pt_1, ev2pt_2

            return ev2pt, ev2pt_1, ev2pt_2

        elif term == 'phi':
            ev_psi, ev_psi1, ev_psi2 = self.ExpectationValue2pt(z, R,
                term='psi', R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)
            ev_oo, ev_oo1, ev_oo2 = self.ExpectationValue2pt(z, R,
                term='oo', R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)

            return ev_psi + ev_oo, ev_psi1 + ev_oo1, ev_psi2 + ev_oo2

        elif term == '21':
            # dTb = T0 * (1 + d21) = T0 * psi
            # d21 = psi - 1
            # <d21 d21'> = <(psi - 1)(psi' - 1)>
            #            = <psi psi'> - 2 <psi> + 1
            if self.pf['ps_include_temp']:
                # New formalism
                # <phi phi'> = <psi psi'> + 2 <psi psi' c> + <psi psi' c c'>
                ev_phi, ev_phi1, ev_phi2 = self.ExpectationValue2pt(z, R,
                    term='phi', R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)

                avg_phi = self.ExpectationValue1pt(z, term='phi',
                    R_s=R_s, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)
                ev21 = ev_phi + 1. - 2. * avg_phi

            else:
                ev_psi, ev_psi1, ev_psi2 = self.ExpectationValue2pt(z, R,
                    term='psi', R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)

                avg_psi = self.ExpectationValue1pt(z, term='psi',
                    R_s=R_s, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)
                ev21 = ev_psi + 1. - 2. * avg_psi


            #raise NotImplemented('still working!')
            #Phi, junk1, junk2 = self.ExpectationValue2pt(z, zeta, R, term='Phi',
            #    R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja, k=k)
            #
            #ev_psi, ev_psi1, ev_psi2 = self.ExpectationValue2pt(z, zeta, R,
            #    term='psi', R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)
            #ev2pt = ev_psi + Phi
            #
            #self._cache_jp_[z][term] = R, ev2pt, Rzeros, Rzeros

            return ev21, Rzeros, Rzeros

        elif term == 'oo':

            # New formalism
            # <phi phi'> = <psi psi'> - 2 <psi psi' c> + <psi psi' c c'>
            #            = <psi psi'> + <o o'>
            ppc, _p1, _p2 = self.ExpectationValue2pt(z, R=R, term='ppc',
                R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)
            ppcc, _p1, _p2 = self.ExpectationValue2pt(z, R=R, term='ppcc',
                R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)

            ev2pt = ppcc - 2 * ppc

            return ev2pt, Rzeros, Rzeros

        #elif term == 'bb':
        #    ev_ii, one, two = self.ExpectationValue2pt(z, zeta, R,
        #        term='ii', R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)
        #    ev_ib, one, two = self.ExpectationValue2pt(z, zeta, R,
        #        term='ib', R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)
        #
        #    if self.pf['ps_include_temp']:
        #        ev_ih, one, two = self.ExpectationValue2pt(z, zeta, R,
        #            term='ih', R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)
        #        ev_hh, one, two = self.ExpectationValue2pt(z, zeta, R,
        #            term='hh', R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)
        #        ev_hb, one, two = self.ExpectationValue2pt(z, zeta, R,
        #            term='hb', R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)
        #    else:
        #        ev_ih = ev_hh = ev_hb = 0.0
        #
        #    #return (1. - Qi - Qh)**2 * Rones, Rzeros, Rzeros
        #
        #    ev_bb = 1. - (ev_ii + 2 * ev_ib + 2 * ev_ih + ev_hh + 2 * ev_hb)
        #
        #    self._cache_jp_[z][term] = R, ev_bb, Rzeros, Rzeros
        #
        #    return ev_bb, Rzeros, Rzeros

        # <psi psi' c> = <cdd'> - 2 <cdi'> - 2 <ci'dd>
        elif term == 'ppc':

            avg_c = self.ExpectationValue1pt(z, term='c',
                R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)
            avg_cd = self.ExpectationValue1pt(z, term='c*d',
                R_s=R_s, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)
            cd, _j1, _j2 = self.ExpectationValue2pt(z, R=R, term='cd',
                R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)
            ci, _j1, _j2 = self.ExpectationValue2pt(z, R=R, term='ic',
                R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)
            cdd, _j1, _j2 = self.ExpectationValue2pt(z, R=R, term='cdd',
                R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)
            cdip, _j1, _j2 = self.ExpectationValue2pt(z, R=R, term='cdip',
                R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)
            cddip, _j1, _j2 = self.ExpectationValue2pt(z, R=R, term='cddip',
                R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)
            cdpip, _j1, _j2 = self.ExpectationValue2pt(z, R=R, term='cdpip',
                R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)

            ppc = avg_c + cd - ci - cdpip + avg_cd + cdd - cdip - cddip

            return ppc, Rzeros, Rzeros

        elif term == 'ppcc':
            ccdd, _j1, _j2 = self.ExpectationValue2pt(z, R=R, term='ccdd',
                R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)
            cc, _j1, _j2 = self.ExpectationValue2pt(z, R=R, term='cc',
                R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)
            ccd, _j1, _j2 = self.ExpectationValue2pt(z, R=R, term='ccd',
                R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)

            return cc + 2 * ccd + ccdd, Rzeros, Rzeros

        elif term in ['mm', 'dd']:
            # Equivalent to correlation function since <d> = 0
            return self.spline_cf_mm(z)(np.log(R)), np.zeros_like(R), np.zeros_like(R)

        #elif term == 'dd':
        #    dd = _P_ii * delta_i_bar**2 \
        #       + _P_hh * delta_h_bar**2 \
        #       + _P_bb * delta_b_bar**2 \
        #       + 2 * _P_ih * delta_i_bar * delta_h_bar \
        #       + 2 * _P_ib * delta_i_bar * delta_b_bar \
        #       + 2 * _P_hb * delta_h_bar * delta_b_bar
        #
        #    return dd, Rzeros, Rzeros

        ##
        # For 3-zone IGM, can compute everything from permutations of
        # i, h, and b.
        ##
        if self.pf['ps_igm_model'] == 1 and not self._getting_basics:
            return self.ThreeZoneModel(z, R=R, term=term,
                R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)



        ##
        # On to things we must be more careful with.
        ##

        # Minimum bubble size
        Mmin = self.Mmin(z) * self.zeta
        iM = np.argmin(np.abs(M_b - Mmin))

        # Only need overlap volumes once per redshift
        all_OV_z = self._cache_Vo(z)
        if all_OV_z is None:
            all_OV_z = np.zeros((len(R), 6, len(R_i)))
            for i, sep in enumerate(R):
                all_OV_z[i,:,:] = \
                    np.array(self.overlap_volumes(sep, R_i, R_s))

            self._cache_Vo_[z] = all_OV_z.copy()

            #print("Generated z={} overlap_volumes".format(z))

        #else:
        #   print("Read in z={} overlap_volumes".format(z))

        all_IV_z = self._cache_IV(z)
        if all_IV_z is None:
            all_IV_z = np.zeros((len(R), 6, len(R_i)))
            for i, sep in enumerate(R):
                all_IV_z[i,:,:] = \
                    np.array(self.intersectional_volumes(sep, R_i, R_s, R3))

            self._cache_IV_[z] = all_IV_z.copy()

        Mmin_b = self.Mmin(z) * self.zeta
        Mmin_h = self.Mmin(z)
        Mmin_s = self.Mmin(z) * zeta_X

        if term in ['hh', 'ih', 'ib', 'hb'] and self.pf['ps_temp_model'] == 1 \
            and self.pf['ps_include_temp']:
            Qh_int = self.get_prob(z, M_s, dndm_s, Mmin, V_h,
                exp=False, ep=0.0, Mmax=None)
            f_h = -np.log(1. - Qh) / Qh_int
        else:
            f_h = 1.

        #dR = np.diff(10**bin_c2e(np.log(R)))#np.concatenate((np.diff(R), [np.diff(R)[-1]]))
        #dR = 10**np.arange(np.log(R).min(), np.log(R).max() + 2 * dlogR, dlogR)

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

            # Yields: V11, V12, V22, V1n, V2n, Van
            # Remember: these radii arrays depend on redshift (through delta_B)
            all_V = all_OV_z[i]
            all_IV = all_IV_z[i]

            # For two-halo terms, need bias of sources.
            if self.pf['ps_include_bias']:
                # Should modify for temp_model==2
                if self.pf['ps_include_temp']:
                    if self.pf['ps_temp_model'] == 2 and 'h' in term:
                        _ion = False
                    else:
                        _ion = True
                else:
                    _ion = True
                ep = self.excess_probability(z, sep, ion=_ion)
            else:
                ep = np.zeros_like(self.m)

            ##
            # For each zone, figure out volume of region where a
            # single source can ionize/heat/couple both points, as well
            # as the region where a single source is not enough (Vss_ne)
            ##
            if term == 'ii':

                Vo = all_V[0]

                # Subtract off more volume if heating is ON.
                #if self.pf['ps_include_temp']:
                #    #Vne1 = Vne2 = V_i - self.IV(sep, R_i, R_s)
                #    Vne1 = Vne2 = V_i - all_IV[1]
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
                Vne1 = Vne2 = V_i - Vo

                _P1 = self.get_prob(z, M_b, dndm_b, Mmin_b, Vo, True)

                _P2_1 = self.get_prob(z, M_b, dndm_b, Mmin_b, Vne1, True)
                _P2_2 = self.get_prob(z, M_b, dndm_b, Mmin_b, Vne2, True, ep)

                _P2 = (1. - _P1) * _P2_1 * _P2_2

                if self.pf['ps_volfix'] and Qi > 0.5:
                    P1[i] = _P1
                    P2[i] = (1. - P1[i]) * _P2_1**2

                else:
                    P1[i] = _P1
                    P2[i] = _P2

            # Probability that one point is ionized, other in "bulk IGM"
            elif term == 'ib':
                Vo_iN = all_V[3] # region in which a source ionized one point
                                 # and does nothing to the other.

                # Probability that a single source does something to
                # each point. If no temp fluctuations, same as _Pis
                P1_iN = self.get_prob(z, M_b, dndm_b, Mmin_b, all_V[3], True)

                # "probability of an ionized pt 2 given ionized pt 1"
                Pigi = self.get_prob(z, M_b, dndm_b, Mmin_b, V_i-all_V[0], True, ep)

                if self.pf['ps_include_temp']:
                    if self.pf['ps_temp_model'] == 1:
                        Vne2 = V_ioh - all_IV[2] - (V_i - all_IV[1])
                        # "probability of a heated pt 2 given ionized pt 1"
                        Phgi = self.get_prob(z, M_b, dndm_b * f_h, Mmin_b, Vne2, True, ep)

                        P2[i] = P1_iN * (1. - Pigi - Phgi)
                    else:
                        P2[i] = Qi * (1. - Qi - Qh)
                else:
                    P2[i] = P1_iN * (1. - Pigi)

            elif term == 'hb':

                #if self.pf['ps_temp_model'] == 2:
                #    print('Ignoring hb term for now...')
                #    continue
                #else:
                #    pass

                if self.pf['ps_temp_model'] == 2:
                    P1_hN = self.get_prob(z, M_s, dndm_s, Mmin_s, all_V[4], True)

                    # Given that the first point is heated, what is the probability
                    # that the second pt is heated or ionized by a different source?
                    # We want the complement of that.

                    # Volume in which I heat but don't ionize (or heat) the other pt,
                    # i.e., same as the two-source term for <hh'>
                    #Vne2 = Vh - self.IV(sep, R_i, R_s)
                    Vne2 = V_ioh - all_IV[2] - (V_i - all_IV[1])

                    # Volume in which single source ioniz
                    V2ii = V_i - all_V[0]

                    Phgh = self.get_prob(z, M_b, dndm_b * f_h, Mmin_b, Vne2, True, ep)
                    Pigh = self.get_prob(z, M_b, dndm_b, Mmin_b, V2ii, True, ep)

                    #P1[i] = P1_hN
                    #ih2 = _P_ih_2[i]
                    #hh2 = _P_hh_2[i]
                    P2[i] = P1_hN * (1. - Phgh - Pigh)
                else:
                    # Probability that single source can heat one pt but
                    # does nothing to the other.
                    P1_hN = self.get_prob(z, M_b, dndm_b * f_h, Mmin_b, all_V[4], True)

                    # Given that the first point is heated, what is the probability
                    # that the second pt is heated or ionized by a different source?
                    # We want the complement of that.

                    # Volume in which I heat but don't ionize (or heat) the other pt,
                    # i.e., same as the two-source term for <hh'>
                    #Vne2 = Vh - self.IV(sep, R_i, R_s)
                    Vne2 = V_ioh - all_IV[2] - (V_i - all_IV[1])

                    # Volume in which single source ioniz
                    V2ii = V_i - all_V[0]

                    Phgh = self.get_prob(z, M_b, dndm_b * f_h, Mmin_b, Vne2, True, ep)
                    Pigh = self.get_prob(z, M_b, dndm_b, Mmin_b, V2ii, True, ep)


                    #P1[i] = P1_hN
                    #ih2 = _P_ih_2[i]
                    #hh2 = _P_hh_2[i]
                    P2[i] = P1_hN * (1. - Phgh - Pigh)


            elif term == 'hh':


                # Excursion set approach for temperature.
                if self.pf['ps_temp_model'] == 2:
                    Vo = all_V[2]

                    Vne1 = Vne2 = V_h - Vo

                    _P1 = self.get_prob(z, M_s, dndm_s, Mmin_s, Vo, True)

                    _P2_1 = self.get_prob(z, M_s, dndm_s, Mmin_s, Vne1, True)
                    _P2_2 = self.get_prob(z, M_s, dndm_s, Mmin_s, Vne2, True, ep)

                    #_P2_1 -= Qi
                    #_P2_1 -= Qi

                    _P2 = (1. - _P1) * _P2_1 * _P2_2

                    #if self.pf['ps_volfix'] and Qi > 0.5:
                    #    P1[i] = _P1
                    #    P2[i] = (1. - P1[i]) * _P2_1**2
                    #
                    #else:
                    P1[i] = _P1
                    P2[i] = _P2

                else:

                    #Vii = all_V[0]
                    #_integrand1 = dndm * Vii
                    #
                    #_exp_int1 = np.exp(-simps(_integrand1[iM:] * M_b[iM:],
                    #    x=np.log(M_b[iM:])))
                    #_P1_ii = (1. - _exp_int1)

                    # Region in which two points are heated by the same source
                    Vo = all_V[2]

                    # Subtract off region of the intersection HH volume
                    # in which source 1 would do *anything* to point 2.
                    #Vss_ne_1 = Vh - (Vo - self.IV(sep, R_i, R_s) + all_V[0])
                    #Vne1 = Vne2 = Vh - Vo
                    # For ionization, this is just Vi - Vo
                    #Vne1 = V2 - all_IV[2] - (V1 - all_IV[1])
                    Vne1 = V_ioh - all_IV[2] - (V_i - all_IV[1])
                    #Vne1 =  V2 - self.IV(sep, R_s, R_s) - (V1 - self.IV(sep, R_i, R_s))
                    Vne2 = Vne1

                    # Shouldn't max(Vo) = Vh?

                    #_P1, _P2 = self.get_prob(z, zeta, Vo, Vne1, Vne2, corr, term)

                    _P1 = self.get_prob(z, M_b, dndm_b * f_h, Mmin_b, Vo, True)

                    _P2_1 = self.get_prob(z, M_b, dndm_b * f_h, Mmin_b, Vne1, True)
                    _P2_2 = self.get_prob(z, M_b, dndm_b * f_h, Mmin_b, Vne2, True, ep)

                    # kludge! to account for Qh = 1 - Qi at late times.
                    # Integrals above will always get over-estimated for hot
                    # regions.
                    #_P2_1 = min(Qh, _P2_1)
                    #_P2_2 = min(Qh, _P2_2)

                    _P2 = (1. - _P1) * _P2_1 * _P2_2

                    # The BSD is normalized so that its integral will recover
                    # zeta * fcoll.

                    # Start chugging along on two-bubble term
                    if np.any(Vne1 < 1e-12):
                        N = sum(Vne1 < 1e-12)
                        print('z={}, R={}: Vss_ne_1 (hh) < 0 {} / {} times'.format(z, sep, N, len(R_s)))
                        #print(Vne1[Vne1 < 1e-12])
                        print(np.all(V_ioh > V_i), np.all(V_ioh > all_IV[2]), all_IV[2][-1], all_IV[1][-1])

                    # Must correct for the fact that Qi+Qh<=1
                    if self.heating_ongoing:
                        P1[i] = _P1
                        P2[i] = _P2
                    else:
                        P1[i] = _P1 * (1. - Qh - Qi)
                        P2[i] = Qh**2

            elif term == 'ih':

                if self.pf['ps_temp_model'] == 2:
                    continue

                if not self.pf['ps_include_xcorr_ion_hot']:
                    P1[i] = 0.0
                    P2[i] = Qh * Qi
                    continue

                #Vo_sh_r1, Vo_sh_r2, Vo_sh_r3 = \
                #    self.overlap_region_shell(sep, R_i, R_s)
                #Vo = 2. * Vo_sh_r2 - Vo_sh_r3
                Vo = all_V[1]

                #V1 = 4. * np.pi * R_i**3 / 3.
                #V2 = 4. * np.pi * R_s**3 / 3.
                #Vh = 4. * np.pi * (R_s**3 - R_i**3) / 3.

                # Volume in which I ionize but don't heat (or ionize) the other pt.
                Vne1 = V_i - all_IV[1]

                # Volume in which I heat but don't ionize (or heat) the other pt,
                # i.e., same as the two-source term for <hh'>
                #Vne2 = Vh - self.IV(sep, R_i, R_s)
                Vne2 =  V_ioh - all_IV[2] - (V_i - all_IV[1])
                #Vne2 =  V2 - self.IV(sep, R_s, R_s) - (V1 - self.IV(sep, R_i, R_s))

                if np.any(Vne2 < 0):
                    N = sum(Vne2 < 0)
                    print('R={}: Vss_ne_2 (ih) < 0 {} / {} times'.format(sep, N, len(R_s)))


                #_P1, _P2 = self.get_prob(z, zeta, Vo, Vne1, Vne2, corr, term)

                _P1 = self.get_prob(z, M_b, dndm_b * f_h, Mmin_b, Vo, True)

                _P2_1 = self.get_prob(z, M_b, dndm_b, Mmin_b, Vne1, True)
                _P2_2 = self.get_prob(z, M_b, dndm_b * f_h, Mmin_b, Vne2, True, ep)

                # Kludge!
                #_P2_1 = min(_P2_2, Qi)
                #_P2_2 = min(_P2_2, Qh)

                _P2 = (1. - _P1) * _P2_1 * _P2_2

                #
                #P2[i] = min(_P2, Qh * Qi)

                if self.heating_ongoing:
                    P1[i] = _P1
                    P2[i] = _P2
                else:
                    P1[i] = _P1 * (1. - Qh - Qi)
                    P2[i] = Qh * Qi

            ##
            # Density stuff from here down
            ##
            if term.count('d') == 0:
                continue


            if not (self.pf['ps_include_xcorr_ion_rho'] \
                 or self.pf['ps_include_xcorr_hot_rho']):
                # These terms will remain zero
                #if term.count('d') > 0:
                continue

            ##
            # First, grab a bunch of stuff we'll need.
            ##

            # Ionization auto-correlations
            #Pii, Pii_1, Pii_2 = \
            #    self.ExpectationValue2pt(z, zeta, R, term='ii',
            #    R_s=R_s, R3=R3, Th=Th, Ts=Ts)

            Vo = all_V[0]
            Vne1 = V_i - Vo

            Vo_hh = all_V[2]

            # These are functions of mass
            Vsh_sph = 4. * np.pi * R_s**3 / 3.
            Vsh = 4. * np.pi * (R_s**3 - R_i**3) / 3.

            # Mean bubble density
            #B = self._B(z, zeta)
            #rho0 = self.cosm.mean_density0
            #delta = M_b / V_i / rho0 - 1.

            M_h = self.halos.tab_M
            iM_h = np.argmin(np.abs(self.Mmin(z) - M_h))
            dndm_h = self.halos.tab_dndm[iz_hmf]
            fcoll_h = self.halos.tab_fcoll[iz_hmf,iM_h]

            # Luminous halos, i.e., Mh > Mmin
            Q_lhal = self.Qhal(z, Mmin=Mmin)
            # Dark halos, i.e., those outside bubbles

            Q_dhal = self.Qhal(z, Mmax=Mmin)
            # All halos
            Q_hal = self.Qhal(z)

            # Since integration must be perfect
            Q_dhal = Q_hal - Q_lhal

            R_hal = self.halos.VirialRadius(M_h, z) / 1e3 # Convert to Mpc
            V_hal = four_pi * R_hal**3 / 3.

            # Bias of bubbles and halos
            ##
            db = self._B(z, ion=True)
            bh = self.halos.Bias(z)
            bb = self.bubble_bias(z, ion=True)
            xi_dd_r = xi_dd[i]#self.spline_cf_mm(z)(np.log(sep))

            bh_bar = self.mean_halo_bias(z)
            bb_bar = self.mean_bubble_bias(z, ion=True)
            ep_bh = bh * bb_bar * xi_dd_r
            exc = bh_bar * bb_bar * xi_dd_r

            # Mean density of halos (mass is arbitrary)
            delta_hal_bar = self.mean_halo_overdensity(z)
            nhal_avg = self.mean_halo_abundance(z)

            # <d> = 0 = <d_i> * Qi + <d_hal> * Q_hal + <d_nohal> * Q_nh
            # Problem is Q_hal and Q_i are not mutually exclusive.
            # Actually: they are inclusive! At least above Mmin.
            delta_nothal_bar = -delta_hal_bar * Q_hal / (1. - Q_hal)

            avg_c = self.ExpectationValue1pt(z, term='c',
                R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)

            #dnh = -dh_avg * fcoll_h / (1. - fcoll_h)

            #_P1_ii = self.get_prob(z, M_b, dndm_b, Mmin_b, Vo, True)
            #delta_i_bar = self.mean_bubble_overdensity(z, zeta)

            if term == 'id':
                ##
                # Analog of one source or one bubble term is P_in, i.e.,
                # probability that points are in the same bubble.
                # The "two bubble term" is instead P_out, i.e., the
                # probability that points are *not* in the same bubble.
                # In the latter case, the density can be anything, while
                # in the former it will be the mean bubble density.
                ##

                # Actually think about halos

                # <x_i d'> = \int d' f(d; x_i=1) f(x_i=1) f(d'; d) dd'
                # Crux is f(d'; d): could be 3-d integral in general.

                # Loop over bubble density.
                #ixd_inner = np.zeros(self.m.size)
                #for k, d1 in enumerate(db):
                #
                #    # Excess probability of halo with mass mh
                #    # given bubble nearby
                #    exc = 0.0#bh * bb[k] * xi_dd_r
                #
                #    #Ph = np.minimum(Ph, 1.)
                #
                #    # <d> = fcoll_V * dh + Qi * di + rest
                #    #Pn = 1. - Ph
                #    #if sep < R_i[k]:
                #    #    dn = d1
                #    #else:
                #    #dn = delta_n_bar
                #
                #    # How to guarantee that <x_i d'> -> 0 on L.S.?
                #    # Is the key formalizing whether we're at the
                #    # center of the bubble or not?
                #    integ = dndm_h * V_h * (1. + exc)
                #
                #    # Don't truncate at Mmin! Don't need star-forming
                #    # galaxy, just need mass.
                #    ixd_inner[k] = np.trapz(integ * M_h, x=np.log(M_h))



                #_integrand = dndm_h * (M_h / rho_bar) * bh
                #fcorr = 1. - np.trapz(_integrand * M_h, x=np.log(M_h))

                # Just halos *outside* bubbles
                hal = np.trapz(dndm_h[:iM_h] * V_hal[:iM_h] * (1. + ep_bh[:iM_h]) * M_h[:iM_h],
                    x=np.log(M_h[:iM_h]))
                bub = np.trapz(dndm_b[iM:] * V_i[iM:] * self.m[iM:],
                    x=np.log(self.m[iM:]))

                P_ihal = (1. - np.exp(-bub)) * (1. - np.exp(-hal))

                #P2[i] = P_ihal * delta_hal_bar + _P_ib[i] * delta_b_bar
                P1[i] = _P_ii_1[i] * delta_i_bar
                P2[i] = _P_ii_2[i] * delta_i_bar + _P_ib[i] * delta_b_bar \
                      + P_ihal * delta_hal_bar

                #P2[i] = Phal * dh_avg + np.exp(-hal) * dnih_avg

                #P2[i] += np.exp(-hal) * dnih_avg * Qi

                #P2[i] += _P_in[i] * delta_n_bar

            elif term in ['cd', 'cdip']:
                continue

            elif term == 'idd':

                hal = np.trapz(dndm_h[:iM_h] * V_hal[:iM_h] * (1. + ep_bh[:iM_h]) * M_h[:iM_h],
                    x=np.log(M_h[:iM_h]))
                bub = np.trapz(dndm_b[iM:] * V_i[iM:] * self.m[iM:],
                    x=np.log(self.m[iM:]))

                P_ihal = (1. - np.exp(-bub)) * (1. - np.exp(-hal))

                #P2[i] = P_ihal * delta_hal_bar + _P_ib[i] * delta_b_bar
                P1[i] = _P_ii_1[i] * delta_i_bar**2
                P2[i] = _P_ii_2[i] * delta_i_bar**2 \
                      + _P_ib[i] * delta_b_bar * delta_i_bar\
                      + P_ihal * delta_hal_bar * delta_i_bar

                #exc = bh_bar * bb_bar * xi_dd_r
                #
                #hal = np.trapz(dndm_h * V_hal * (1. + exc) * M_h,
                #    x=np.log(M_h))
                #bub = np.trapz(dndm_b[iM:] * V_i[iM:] * self.m[iM:],
                #    x=np.log(self.m[iM:]))
                #
                #P2[i] = ((1. - np.exp(-hal)) * delta_hal_bar
                #      + np.exp(-hal) * delta_nothal_bar) \
                #      * (1. - np.exp(-bub)) * delta_i_bar



                #_P1 = _P_ii_1[i] * delta_i_bar**2
                #
                ##_P2 = _P_ii_2[i] * delta_i_bar**2 \
                ##    + Qi * (1. - _P_ii_1[i]) * delta_i_bar * delta_n_bar \
                ##    - Qi**2 * delta_i_bar**2
                ##
                #P1[i] = _P1
                #
                ##P2[i] = Qi * xi_dd[i] # There's gonna be a bias here
                ##P2[i] = _P_ii_2[i] * delta_i_bar**2 - Qi**2 * delta_i_bar**2
                #
                ##continue
                #
                #idd_ii = np.zeros(self.m.size)
                #idd_in = np.zeros(self.m.size)
                #
                ## Convert from dm to dd
                ##dmdd = np.diff(self.m) / np.diff(db)
                ##dmdd = np.concatenate(([0], dmdd))
                #
                ## Should be able to speed this up
                #
                #for k, d1 in enumerate(db):
                #
                #
                #    exc = bb[k] * bb * xi_dd_r
                #
                #    grand = db[k] * dndm_b[k] * V_i[k] \
                #          * db * dndm_b * V_i \
                #          * (1. + exc)
                #
                #    idd_ii[k] = np.trapz(grand[iM:] * self.m[iM:],
                #        x=np.log(self.m[iM:]))
                #
                #    #exc_in = bb[k] * bh * xi_dd_r
                #    #
                #    #grand_in = db[k] * dndm_b[k] * V_i[k] #\
                #    #      #* dh * dndm_h * Vvir \
                #    #      #* (1. + exc_in)
                #    #
                #    #idd_in[k] = np.trapz(grand_in[iM_h:] * M_h[iM_h:],
                #    #    x=np.log(M_h[iM_h:]))
                #
                ##idd_in = np.trapz(db[iM:] * dndm_b[iM:] * V_i[iM:] * delta_n_bar * self.m[iM:],
                ##    x=np.log(self.m[iM:]))
                #
                #
                #P2[i] = _P_ii_2[i]  \
                #    * np.trapz(idd_ii[iM:] * self.m[iM:],
                #        x=np.log(self.m[iM:]))
                #
                ## Another term for <x_i x'> possibility. Doesn't really
                ## fall into the one bubble two bubble language, so just
                ## sticking it in P2.
                ## Assumes neutral phase is at cosmic mean neutral density
                ## Could generalize...
                #P2[i] += _P_ib[i] * delta_i_bar * delta_b_bar
                #
                ## We're neglecting overdensities by just using the
                ## mean neutral density
                #
                #continue

            elif term == 'iid':

                # This is like the 'id' term except the second point
                # has to be ionized.
                P2[i] = _P_ii[i] * delta_i_bar
            elif term == 'iidd':

                P2[i] = _P_ii[i] * delta_i_bar**2

                continue

                # Might have to actually do a double integral here.

                iidd_2 = np.zeros(self.m.size)

                # Convert from dm to dd
                #dmdd = np.diff(self.m) / np.diff(db)
                #dmdd = np.concatenate(([0], dmdd))

                # Should be able to speed this up

                for k, d1 in enumerate(db):
                    exc = bb[k] * bb * xi_dd_r

                    grand = db[k] * dndm_b[k] * V_i[k] \
                          * db * dndm_b * V_i \
                          * (1. + exc)

                    iidd_2[k] = np.trapz(grand[iM:] * self.m[iM:],
                        x=np.log(self.m[iM:]))

                P2[i] = _P_ii_2[i]  \
                    * np.trapz(iidd_2[iM:] * self.m[iM:],
                        x=np.log(self.m[iM:]))

            #elif term == 'cd':
            #
            #    if self.pf['ps_include_xcorr_hot_rho'] == 0:
            #        break
            #    elif self.pf['ps_include_xcorr_hot_rho'] == 1:
            #        hal = np.trapz(dndm_h * V_hal * (1. + exc) * M_h,
            #            x=np.log(M_h))
            #        hot = np.trapz(dndm_b[iM:] * V_h[iM:] * self.m[iM:],
            #            x=np.log(self.m[iM:]))
            #        P2[i] = ((1. - np.exp(-hal)) * dh_avg + np.exp(-hal) * dnih_avg) \
            #          * (1. - np.exp(-hot)) * avg_c
            #    elif self.pf['ps_include_xcorr_hot_rho'] == 2:
            #        P2[i] = _P_hh[i] * delta_h_bar + _P_hb[i] * delta_b_bar
            #    else:
            #        raise NotImplemented('help')
            #
            #elif term == 'ccd':
            #
            #
            #    _P1 = _P_hh_1[i] * delta_i_bar
            #    #B = self._B(z, zeta)
            #    #_P1 = delta_i_bar \
            #     #   * self.get_prob(z, M_b, dndm_b, Mmin_b, Vo, True) \
            #
            #
            #    _P2 = _P_hh_2[i] * delta_i_bar
            #
            #    #_P2 = (1. - _P_ii_1[i]) * delta_i_bar \
            #    #    * self.get_prob(z, M_b, dndm_b, Mmin_b, Vne1, True) \
            #    #    * self.get_prob(z, M_b, dndm_b, Mmin_b, Vne1, True, ep)
            #    #    #* self.get_prob(z, M_h, dndm_h, Mmi\n_h, Vvir, False, ep_bb)
            #
            #    P1[i] = _P1
            #    P2[i] = _P2
            #
            elif term == 'cdd':

                raise NotImplemented('help')
                hal = np.trapz(dndm_h * V_hal * (1. + exc) * M_h,
                    x=np.log(M_h))
                hot = np.trapz(dndm_b[iM:] * Vsh[iM:] * self.m[iM:],
                    x=np.log(self.m[iM:]))
                hoi = np.trapz(dndm_b[iM:] * Vsh_sph[iM:] * self.m[iM:],
                    x=np.log(self.m[iM:]))    # 'hot or ionized'
                # One point in shell, other point in halo
                # One point ionized, other point in halo
                # Q. Halos exist only in ionized regions?

            elif term == 'ccdd':
                raise NotImplemented('help')

                # Add in bulk IGM correction.
                # Add routines for calculating 'ib' and 'hb'?
                P2[i] = _P_hh[i] * delta_h_bar**2 \
                      + _P_bb[i] * delta_h_bar**2 \
                      + _P_hb[i] * delta_h_bar * delta_b_bar
                continue

                # Might have to actually do a double integral here.

                iidd_2 = np.zeros(self.m.size)

                # Convert from dm to dd
                #dmdd = np.diff(self.m) / np.diff(db)
                #dmdd = np.concatenate(([0], dmdd))

                # Should be able to speed this up

                for k, d1 in enumerate(db):
                    exc = bb[k] * bb * xi_dd_r

                    grand = db[k] * dndm_b[k] * V_i[k] \
                          * db * dndm_b * V_i \
                          * (1. + exc)

                    iidd_2[k] = np.trapz(grand[iM:] * self.m[iM:],
                        x=np.log(self.m[iM:]))

                P2[i] = _P_ii_2[i]  \
                    * np.trapz(iidd_2[iM:] * self.m[iM:],
                        x=np.log(self.m[iM:]))

            else:
                raise NotImplementedError('No method found for term=\'{}\''.format(term))

        ##
        # SUM UP
        ##
        PT = P1 + P2

        if term in ['ii', 'hh', 'ih', 'ib', 'hb', 'bb']:
            if term not in self._cache_jp_[z]:
                self._cache_jp_[z][term] = R, PT, P1, P2

        return PT, P1, P2

    def ThreeZoneModel(self, z, R, term='ii', R_s=None, R3=None,
        Th=500.0, Ts=None, Tk=None, Ja=None, k=None):
        """
        Model in which IGM partitioned into three phases: ionized, hot, bulk.

        .. note :: If ps_include_temp==False, this is just a two-zone model
            since there is no heated phase in this limit.
        """

        if not self._getting_basics:
            basics = self.get_basics(z, R, R_s=R_s, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)

            if term in basics:
                return basics[term]

            _P_ii, _P_ii_1, _P_ii_2 = basics['ii']
            _P_hh, _P_hh_1, _P_hh_2 = basics['hh']
            _P_bb, _P_bb_1, _P_bb_2 = basics['bb']
            _P_ih, _P_ih_1, _P_ih_2 = basics['ih']
            _P_ib, _P_ib_1, _P_ib_2 = basics['ib']
            _P_hb, _P_hb_1, _P_hb_2 = basics['hb']

        Rones  = np.zeros_like(R)
        Rzeros = np.zeros_like(R)
        delta_i_bar = self.delta_bubble_vol_weighted(z, ion=True)
        delta_h_bar = self.delta_shell(z)
        delta_b_bar = self.BulkDensity(z, R_s)
        Qi = self.MeanIonizedFraction(z)
        Qh = self.MeanIonizedFraction(z, ion=False)

        Tcmb = self.cosm.TCMB(z)
        ci = 0.0
        ch = self.TempToContrast(z, Th=Th, Tk=Tk, Ts=Ts, Ja=Ja)
        if Ts is None:
            cb = 0.0
        else:
            cb = Tcmb / Ts

        ##
        # On to derived quantities
        ##
        if term == 'cc':

            result = Rzeros.copy()
            if self.pf['ps_include_temp']:
                result += _P_hh * ch**2 + 2 * _P_hb * ch * cb + _P_bb * cb**2

            if self.pf['ps_include_lya']:
                xa = self.hydr.RadiativeCouplingCoefficient(z, Ja, Tk)

                if xa < self.pf['ps_lya_cut']:
                    ev_aa, ev_aa1, ev_aa2 = \
                        self.ExpectationValue2pt(z, R, term='aa',
                            R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, k=k, Ja=Ja)

                    Tcmb = self.cosm.TCMB(z)
                    result += ev_aa / (1. + xa)**2

            return result, Rzeros, Rzeros

        elif term == 'ic':
            if not self.pf['ps_include_xcorr_ion_hot']:
                return (Qi**2 * ci + Qh * Qi * ch) * Rones, Rzeros, Rzeros

            ev = _P_ih * ch + _P_ib * cb

            return ev, Rzeros, Rzeros

        elif term == 'icc':
            ch = self.TempToContrast(z, Th=Th, Tk=Tk, Ts=Ts, Ja=Ja)
            ci = self.BubbleContrast(z, Th=Th, Tk=Tk, Ts=Ts, Ja=Ja)
            ev_ii, ev_ii1, ev_ii2 = self.ExpectationValue2pt(z, R=R,
                term='ii', R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)
            ev_ih, ev_ih1, ev_ih2 = self.ExpectationValue2pt(z, R=R,
                term='ih', R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)

            return ev_ii * ci**2 + ev_ih * ch * ci, Rzeros, Rzeros

        elif term == 'iidd':

            if self.pf['ps_use_wick']:
                ev_ii, one_pt, two_pt = self.ExpectationValue2pt(z, R,
                    term='ii')
                ev_dd, one_pt, two_pt = self.ExpectationValue2pt(z, R,
                    term='dd')
                ev_id, one_pt, two_pt = self.ExpectationValue2pt(z, R,
                    term='id')

                #if self.pf['ps_include_ion']:
                avg_id = self.ExpectationValue1pt(z, term='i*d')

                idt, id1, id2 = \
                    self.ExpectationValue2pt(z, R, term='id')

                return ev_ii * ev_dd + ev_id**2 + avg_id**2, Rzeros, Rzeros
            else:
                return _P_ii * delta_i_bar**2, Rzeros, Rzeros

        elif term == 'id':
            P = _P_ii * delta_i_bar \
              + _P_ih * delta_h_bar \
              + _P_ib * delta_b_bar
            return P, Rzeros, Rzeros
        elif term == 'iid':
            return _P_ii * delta_i_bar, Rzeros, Rzeros
        elif term == 'idd':
            P = _P_ii * delta_i_bar**2 \
              + _P_ih * delta_i_bar * delta_h_bar \
              + _P_ib * delta_i_bar * delta_b_bar
            return P, Rzeros, Rzeros
        elif term == 'cd':
            if not self.pf['ps_include_xcorr_hot_rho']:
                return Rzeros, Rzeros, Rzeros

            ev = _P_ih * ch * delta_i_bar \
               + _P_ib * cb * delta_i_bar \
               + _P_hh * ch * delta_h_bar \
               + _P_hb * ch * delta_b_bar \
               + _P_hb * cb * delta_h_bar \
               + _P_bb * cb * delta_b_bar

            return ev, Rzeros, Rzeros

        elif term == 'ccdd' and self.pf['ps_use_wick']:

            ev_cc, one_pt, two_pt = self.ExpectationValue2pt(z, R,
                term='cc')
            ev_dd, one_pt, two_pt = self.ExpectationValue2pt(z, R,
                term='dd')
            ev_cd, one_pt, two_pt = self.ExpectationValue2pt(z, R,
                term='cd')

            #if self.pf['ps_include_ion']:
            avg_cd = self.ExpectationValue1pt(z, term='c*d')

            cdt, cd1, cd2 = \
                self.ExpectationValue2pt(z, R, term='cd')

            return ev_cc * ev_dd + ev_cd**2 + avg_cd**2, Rzeros, Rzeros

        elif term == 'aa':
            aa = self.CorrelationFunction(z, R, term='aa',
                R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja, k=k)

            return aa, Rzeros, Rzeros

        ##
        # BUNCHA TEMPERATURE-DENSITY STUFF BELOW
        ##
        elif term == 'ccd':
            ev = _P_hh * ch**2 * delta_h_bar \
               + _P_hb * ch * cb * delta_h_bar \
               + _P_hb * ch * cb * delta_b_bar \
               + _P_bb * cb**2 * delta_b_bar

            return ev, Rzeros, Rzeros
        elif term == 'cdd':
            ev = _P_ih * ch * delta_i_bar * delta_h_bar \
               + _P_ib * cb * delta_i_bar * delta_b_bar \
               + _P_hh * ch * delta_h_bar**2 \
               + _P_hb * ch * delta_h_bar * delta_b_bar \
               + _P_hb * cb * delta_h_bar * delta_b_bar \
               + _P_bb * cb * delta_b_bar**2

            return ev, Rzeros, Rzeros

        # <c d x_i'>
        elif term == 'cdip':
            ev = _P_ih * delta_h_bar * ch + _P_ib * delta_b_bar * cb

            return ev, Rzeros, Rzeros

        # <c d d' x_i'>
        elif term == 'cddip':
            ev = _P_ih * delta_i_bar * delta_h_bar * ch \
               + _P_ib * delta_i_bar * delta_b_bar * cb

            return ev, Rzeros, Rzeros

        elif term == 'cdpip':
            ev = _P_ih * delta_i_bar * ch \
               + _P_ib * delta_i_bar * cb

            return ev, Rzeros, Rzeros

        # Wick's theorem approach above
        elif term == 'ccdd':
            ev = _P_hh * delta_h_bar**2 * ch**2 \
               + 2 * _P_hb * delta_h_bar * ch * delta_b_bar * cb \
               + _P_bb * delta_b_bar**2 * cb**2

            return ev, Rzeros, Rzeros

        else:
            raise NotImplementedError('No model for term={} in ThreeZoneModel.'.format(term))


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

        integr = np.trapz(integrand[iM:iM2] * M[iM:iM2], x=np.log(M[iM:iM2]))

        # Exponentiate?
        if exp:
            exp_int = np.exp(-integr)
            P = 1. - exp_int
        else:
            P = integr

        return P

    def CorrelationFunction(self, z, R=None, term='ii',
        R_s=None, R3=0.0, Th=500., Tc=1., Ts=None, k=None, Tk=None, Ja=None):
        """
        Compute the correlation function of some general term.

        """

        Qi = self.MeanIonizedFraction(z)
        Qh = self.MeanIonizedFraction(z, ion=False)

        if R is None:
            use_R_tab = True
            R = self.halos.tab_R
        else:
            use_R_tab = False

        if Qi == 1:
            return np.zeros_like(R)

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
                self.ExpectationValue2pt(z, R=R, term='psi',
                    R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)
            avg_psi = self.ExpectationValue1pt(z, term='psi',
                R_s=R_s, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)

            cf_psi = ev_2pt - avg_psi**2

            ##
            # Temperature fluctuations
            ##
            include_temp = self.pf['ps_include_temp']
            include_lya =  self.pf['ps_include_lya']
            if (include_temp or include_lya) and term in ['phi', '21']:

                ev_oo, o1, o2 = self.ExpectationValue2pt(z, R=R,
                    term='oo', R_s=R_s, Ts=Ts, Tk=Tk, Th=Th, Ja=Ja, k=k)
                avg_oo = self.ExpectationValue1pt(z, term='oo',
                    R_s=R_s, Ts=Ts, Tk=Tk, Th=Th, Ja=Ja, R3=R3)

                cf_omega = ev_oo - avg_oo

                cf_21 = cf_psi + cf_omega # i.e., cf_phi

            else:
                cf_21 = cf_psi

            if term == '21':
                cf = cf_21
            elif term == 'phi':
                cf = cf_21
            elif term == 'psi':
                cf = cf_psi

        elif term == 'nn':
            cf = -self.CorrelationFunction(z, R, term='ii',
                R_s=R_s, R3=R3, Th=Th, Ts=Ts)
            return cf
        elif term == 'nc':
            cf = -self.CorrelationFunction(z, R, term='ic',
                R_s=R_s, R3=R3, Th=Th, Ts=Ts)
            return cf
        elif term == 'nd':
            cf = -self.CorrelationFunction(z, R, term='id',
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
                self.ExpectationValue2pt(z, R=R, term='ii',
                    R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)

            ev_i = self.ExpectationValue1pt(z, term='i',
                R_s=R_s, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)
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
                self.ExpectationValue2pt(z, R=R, term='hh',
                    R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)

            #if self.pf['ps_volfix']:
            #    if (Qh < 0.5):
            #        ev_hh = jp_hh_1 + jp_hh_2
            #    else:
            #        # Should this 1-Qh factor be 1-Qh-Qi?
            #        ev_hh = (1. - Qh) * jp_hh_1 + Qh**2
            #else:
            #    ev_hh = jp_hh

            # Should just be Qh
            ev_h = self.ExpectationValue1pt(z, term='h',
                R_s=R_s, Ts=Ts, Th=Th, Tk=Tk, Ja=Ja)

            cf = jp_hh - ev_h**2

        ##
        # Ionization-density cross correlation function
        ##
        elif term == 'id':
            if self.pf['ps_include_xcorr_ion_rho'] == 0:
                cf = np.zeros_like(R)
                self._cache_cf_[z][term] = R, cf
                return cf

            #jp_ii, jp_ii_1, jp_ii_2 = \
            #    self.ExpectationValue2pt(z, R, term='ii',
            #    R_s=R_s, R3=R3, Th=Th, Ts=Ts)

            jp_im, jp_im_1, jp_im_2 = \
                self.ExpectationValue2pt(z, R, term='id',
                R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)

            ev_xd = 0.0

            # Add optional correction to ensure limiting behavior?
            if self.pf['ps_volfix']:
                #if Qi < 0.5:
                ev = jp_im
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
                self.ExpectationValue2pt(z, R=R, term='cc',
                    R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)

            ev_c = self.ExpectationValue1pt(z, term='c',
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

            if self.pf['ps_temp_model'] == 2:
                return np.zeros_like(R)

            ev_2pt, ev_1, ev_2 = \
                self.ExpectationValue2pt(z, R=R, term='ih',
                    R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)

            # Add optional correction to ensure limiting behavior?
            #if self.pf['ps_volfix']:
            #    if (Qh < 0.5) and (Qi < 0.5):
            #        ev_2pt = jp_1 + jp_2
            #    else:
            #        ev_2pt = (1. - Qh) * jp_1 + Qh * Qi
            #else:
            #    ev_2pt = jp + Qh * Qi

            ev_1pt_i = self.ExpectationValue1pt(z, term='i',
                R_s=R_s, Ts=Ts, Th=Th, Tk=Tk, Ja=Ja)
            ev_1pt_h = self.ExpectationValue1pt(z, term='h',
                R_s=R_s, Ts=Ts, Th=Th, Tk=Tk, Ja=Ja)

            cf = ev_2pt - ev_1pt_i * ev_1pt_h

        elif term == 'ic':
            ev_2pt, ev_1, ev_2 = \
                self.ExpectationValue2pt(z, R=R, term='ic',
                    R_s=R_s, R3=R3, Th=Th, Ts=Ts, Tk=Tk, Ja=Ja)

            # Add optional correction to ensure limiting behavior?
            #if self.pf['ps_volfix']:
            #    if (Qh < 0.5) and (Qi < 0.5):
            #        ev_2pt = jp_1 + jp_2
            #    else:
            #        ev_2pt = (1. - Qh) * jp_1 + Qh * Qi
            #else:
            #    ev_2pt = jp + Qh * Qi

            ev_1pt = self.ExpectationValue1pt(z, term='i*c',
                R_s=R_s, Ts=Ts, Th=Th, Tk=Tk, Ja=Ja)

            # Don't square the second term!
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
            if self.pf['ps_include_lya_lc'] == False:
                a = None
            elif type(self.pf['ps_include_lya_lc']) is float:
                a = lambda zz: self.pf['ps_include_lya_lc']
            else:
                # Use specific mass accretion rate of Mmin halo
                # to get characteristic halo growth time. This is basically
                # independent of mass so it should be OK to just pick Mmin.

                #oot = lambda zz: self.pops[0].dfcolldt(z) / self.pops[0].halos.fcoll_2d(zz, np.log10(Mmin(zz)))
                #a = lambda zz: (1. / oot(zz)) / pop.cosm.HubbleTime(zz)
                oot = lambda zz: self.halos.MAR_func(zz, Mmin(zz)) / Mmin(zz) / s_per_yr
                a = lambda zz: (1. / oot(zz)) / self.cosm.HubbleTime(zz)

            if a is not None:
                tstar = lambda zz: a(zz) * self.cosm.HubbleTime(zz)
                rstar = c * tstar(z) * (1. + z) / cm_per_mpc
                ulya = lambda kk, mm, zz: self.halos.u_isl_exp(kk, mm, zz, rmax, rstar)
            else:
                ulya = lambda kk, mm, zz: self.halos.u_isl(kk, mm, zz, rmax)

            ps_try = self._cache_ps(z, 'aa')

            if ps_try is not None:
                ps = ps_try
            else:
                ps = np.array([self.halos.PowerSpectrum(z, _k, ulya, Mmin(z)) \
                    for _k in k])
                self._cache_ps_[z][term] = ps

            cf = self.CorrelationFunctionFromPS(R, ps, k, split_by_scale=True)

        else:
            raise NotImplementedError('Unrecognized correlation function: {}'.format(term))

        #if term not in ['21', 'mm']:
        #    cf /= (2. * np.pi)**3

        self._cache_cf_[z][term] = R, cf.copy()

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

    def BubbleContrast(self, z, Th=500., Tk=None, Ts=None, Ja=None):
        return 0.0

        Tcmb = self.cosm.TCMB(z)
        Tgas = self.cosm.Tgas(z)

        if Tk is None:
            print("Assuming Tk=Tgas(unheated).")
            Tk = Tgas

        if Ts is None:
            print("Assuming Ts=Tk.")
            Ts = Tk

        if Ja is None:
            print("Assuming xa=0.")
            xa = Ja = 0.0
        else:
            xa = self.hydr.RadiativeCouplingCoefficient(z, Ja, Tk)

        return 0.0#Tcmb / (Ts - Tcmb)

    def TempToContrast(self, z, Th=500., Tk=None, Ts=None, Ja=None):
        """
        Find value of 'contrast' fluctuation given mean temperature Tk and
        assumed kinetic temperature of heated regions, Th.
        """

        if self.pf['ps_temp_model'] == 2:
            return 1. / self.pf['ps_saturated']

        if Th is None:
            return 0.0

        Tcmb = self.cosm.TCMB(z)
        Tgas = self.cosm.Tgas(z)

        if Tk is None:
            print("Assuming Tk=Tgas(unheated).")
            Tk = Tgas

        if Ts is None:
            print("Assuming Ts=Tk.")
            Ts = Tk

        # Don't let heated regions be colder than cosmic mean temperature.
        # Forces delta_T -> 0 at late times
        Th = max(Th, Tk)

        delta_T = Th / Tk - 1.


        # NEW CONTRAST DEFINITION
        return Tcmb / Th


        #if ii <= 1:

        # Contrast of hot regions.
        #return (1. - Tcmb / Th) / (1. - Tcmb / Ts) - 1.
        #return (delta_T / (1. + delta_T)) * (Tcmb / (Tk - Tcmb))

    def CorrelationFunctionFromPS(self, R, ps, k=None, split_by_scale=False,
        kmin=None, epsrel=1-8, epsabs=1e-8, method='clenshaw-curtis',
        use_pb=False, suppression=np.inf):

        if np.all(ps == 0):
            return np.zeros_like(R)

        return self.halos.InverseFT3D(R, ps, k, kmin=kmin,
            epsrel=epsrel, epsabs=epsabs, use_pb=use_pb,
            split_by_scale=split_by_scale, method=method, suppression=suppression)

    def PowerSpectrumFromCF(self, k, cf, R=None, split_by_scale=False,
        Rmin=None, epsrel=1-8, epsabs=1e-8, method='clenshaw-curtis',
        use_pb=False, suppression=np.inf):

        if np.all(cf == 0):
            return np.zeros_like(k)

        return self.halos.FT3D(k, cf, R, Rmin=Rmin,
            epsrel=epsrel, epsabs=epsabs, use_pb=use_pb,
            split_by_scale=split_by_scale, method=method, suppression=suppression)
