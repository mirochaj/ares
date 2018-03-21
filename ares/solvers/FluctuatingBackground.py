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
from scipy.optimize import fsolve
from ..util.Math import LinearNDInterpolator
from ..populations.Composite import CompositePopulation
from ..physics.CrossSections import PhotoIonizationCrossSection
from ..physics.Constants import g_per_msun, cm_per_mpc, dnu, s_per_yr, c, \
    s_per_myr, erg_per_ev, k_B, m_p, dnu

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
            iM = np.argmin(np.abs(Mmin * zeta - self.halos.M))

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
            
            if self.pf['powspec_rescale_Qion'] and self.pf['powspec_rescale_Qhot']:
                norm = min(zeta * self.halos.fcoll_2d(z, np.log10(self.Mmin(z))), 1)
                
                corr = (norm / Qi)
                Qh *= corr

        else:
            raise NotImplemented('Uncrecognized option for BSD.')
         
        return min(Qh, 1.), min(Qc, 1.)

    def BubbleFillingFactor(self, z, zeta, zeta_lya=None, lya=False):

        if self.pf['bubble_size_dist'] is None:
            Ri = self.pf['bubble_size']
            Vi = 4. * np.pi * R_b**3 / 3.
            ni = self.BubbleDensity(z)

            Qi = 1. - np.exp(-ni * Vi)

        elif self.pf['bubble_size_dist'].lower() == 'fzh04':

            # Smallest bubble is one around smallest halo.
            Mmin = self.Mmin(z)
            iM = np.argmin(np.abs(Mmin * zeta - self.halos.M))

            Ri, Mi, dndm = self.BubbleSizeDistribution(z, zeta)
            Vi = 4. * np.pi * Ri**3 / 3.

            dndlnm = dndm * Mi

            if self.pf['powspec_rescale_Qion']:
                Qi = min(zeta * self.halos.fcoll_2d(z, np.log10(Mmin)), 1)
            else:
                Qi = np.trapz(dndlnm[iM:] * Vi[iM:], x=np.log(Mi[iM:]))
            
        else:
            raise NotImplemented('Uncrecognized option for BSD.')
        
        # Grab heated phase to enforce BC
        #Rs = self.BubbleShellRadius(z, Ri)        
        #Vsh = 4. * np.pi * (Rs - Ri)**3 / 3.
        #Qh = np.trapz(dndm * Vsh * Mi, x=np.log(Mi))   
        
        if lya and self.pf['bubble_pod_size_func'] in [None, 'const', 'linear']:
            Rc = self.BubblePodRadius(z, Ri, zeta, zeta_lya)
            Vc = 4. * np.pi * (Rc - Ri)**3 / 3.
            
            if self.pf['powspec_rescale_Qlya']:
                # This isn't actually correct since we care about fluxes
                # not number of photons, but fine for now.
                Qc = min(zeta_lya * self.halos.fcoll_2d(z, np.log10(self.Mmin(z))), 1)
            else:
                Qc = np.trapz(dndlnm[iM:] * Vc[iM:], x=np.log(Mi[iM:]))

            return min(Qc, 1.)

        elif lya and self.pf['bubble_pod_size_func'] == 'fzh04':
            return self.BubbleFillingFactor(z, zeta_lya, None, lya=False)
        else:
            return min(Qi, 1.)

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

                try:
                    Mmin_tab = np.minimum(Mmin_tab, pop._tab_Mmin)
                except AttributeError:
                    Mmin_tab = np.minimum(Mmin_tab, 10**pop.halos.logM_min)
            
            self._Mmin = lambda z: np.interp(z, self.pops[0].halos.z, Mmin_tab)
        
        return self._Mmin
                
    def mean_bubble_bias(self, z, zeta, zeta_lya, term):
        Ri, Mi, dndm = self.BubbleSizeDistribution(z, zeta)
                
        if ('h' in term) or ('c' in term) and self.pf['powspec_temp_method'] == 'shell':
            Rh, Rc = self.BubbleShellRadius(z, Ri)
            R = Rh
        else:
            R = Ri
        
        V = 4. * np.pi * R**3 / 3.
        
        Mmin = self.Mmin(z) * zeta
        iM = np.argmin(np.abs(Mmin - self.halos.M))
        bHII = self.bubble_bias(z, zeta)

        return np.trapz(dndm[iM:] * V[iM:] * bHII[iM:] * Mi[iM:],
            x=np.log(Mi[iM:]))

    def bubble_bias(self, z, zeta):
        iz = np.argmin(np.abs(z - self.halos.z))
        s = self.halos.sigma_0 #* self.halos.growth_factor[iz]

        Mmin = self.Mmin(z) * zeta

        #return 1. + self._B0(z, zeta)**2 / s**2 \
        #    / self.LinearBarrier(z, zeta, zeta * Mmin)
        return 1. + ((self._B(z, zeta, zeta)**2 / s**2 - (1. / self._B0(z, zeta))) \
            / self.halos.growth_factor[iz])

    def excess_probability(self, z, r, data, zeta, zeta_lya, term='ii'):
        """
        This is the excess probability that a point is ionized given that 
        we already know another point (at distance r) is ionized.
        """

        if term == 'ii':
            Q = data['Qi']
        elif term == 'cc':
            Q = data['Qc']
        else: 
            Q = data['Qh']

        bHII = self.bubble_bias(z, zeta)
        bbar = self.mean_bubble_bias(z, zeta, zeta_lya, term) / Q

        # Should there be a baryon fraction multiplier here?
        xi_dd = np.interp(np.log(r), np.log(data['dr']), data['cf_dd'])

        return np.maximum(bHII * bbar * xi_dd, 0.0)

    def _K(self, zeta):
        return erfinv(1. - 1. / zeta)
    
    def _delta_c(self, z, popid=0):
        pop = self.pops[popid]
        return pop.cosm.delta_c0 / pop.growth_factor(z)

    def _B0(self, z, zeta=40.):

        iz = np.argmin(np.abs(z - self.halos.z))
        s = self.halos.sigma_0 #* self.halos.growth_factor[iz]

        Mmin = self.Mmin(z)
        
        # Variance on scale of smallest collapsed object
        sigma_min = np.interp(Mmin * zeta, self.halos.M, s)
        return self._delta_c(z) - np.sqrt(2.) * self._K(zeta) * sigma_min
    
    def _B1(self, z, zeta=40.):
        iz = np.argmin(np.abs(z - self.halos.z))
        s = self.halos.sigma_0 #* self.halos.growth_factor[iz]
        
        Mmin = self.Mmin(z)
        sigma_min = np.interp(Mmin * zeta, self.halos.M, s)
        ddx_ds2 = self._K(zeta) / np.sqrt(2. * (sigma_min**2 - s**2))
    
        return ddx_ds2[s == s.min()]
    
    def _B(self, z, zeta, zeta_min):
        return self.LinearBarrier(z, zeta, zeta_min)
        
    def LinearBarrier(self, z, zeta, zeta_min):
        s = self.halos.sigma_0
        
        return self._B0(z, zeta_min) + self._B1(z, zeta) * s**2
    
    def Barrier(self, z, zeta, zeta_min):
        """
        Full barrier.
        """

        Mmin = self.Mmin(z)
        s = self.halos.sigma_0
        sigma_min = np.interp(Mmin * zeta, self.halos.M, s)

        return self._delta_c(z) - np.sqrt(2.) * self._K(zeta) \
            * np.sqrt(sigma_min**2 - s**2)

    def BubblePodSizeDistribution(self, z, zeta):
        if self.pf['powspec_lya_method'] == 1:
            # Need to modify zeta and critical threshold
            Rc, Mc, dndm = self.BubbleSizeDistribution(z, zeta)
            return Rc, Mc, dndm
        else:
            raise NotImplemented('help please')

    def BubbleSizeDistribution(self, z, zeta):

        #if not hasattr(self, '_bsd_cache'):
        #    self._bsd_cache = {}

        #if z in self._bsd_cache:
        #    Ri, Mi, dndm = self._bsd_cache[(z,lya)]
           
        if self.pf['bubble_size_dist'] is None:
            if self.pf['bubble_density'] is not None:
                Ri = self.pf['bubble_size']
                Mi = (4. * np.pi * Rb**3 / 3.) * self.cosm.mean_density0
                dndm = self.pf['bubble_density']
            else:
                raise NotImplementedError('help')
        
        elif self.pf['bubble_size_dist'].lower() == 'fzh04':
            Mi = self.halos.M
            rho0 = self.cosm.mean_density0
            delta_B = self._B(z, zeta, zeta)
            Ri = ((Mi / rho0 / (1. + delta_B)) * 0.75 / np.pi)**(1./3.)
        
            iz = np.argmin(np.abs(z - self.halos.z))
            sig = self.halos.sigma_0 #* self.halos.growth_factor[iz]
        
            S = sig**2
        
            Mmin = self.Mmin(z) #* zeta # doesn't matter for zeta=const
            if type(zeta) == np.ndarray:
                zeta_min = np.interp(Mmin, self.halos.M, zeta)
            else:
                zeta_min = zeta
        
            # Shouldn't there be a correction factor here to account for the
            # fact that some of the mass is He?
        
            pcross = self._B0(z, zeta_min) / np.sqrt(2. * np.pi * S**3) \
                * np.exp(-0.5 * self.LinearBarrier(z, zeta, zeta_min)**2 / S)
        
            # This is Eq. 9.38 from Steve's book.
            # The factor of 2 is from using dlns instead of dS (where S=s^2)
            dndm = rho0 * pcross * 2 * np.abs(self.halos.dlns_dlnm) \
                * S / Mi**2
             
            # Not sure I understand how this works. Infinitely recursive at this
            # point.
            #if self.pf['powspec_rescale_Qion']:
            #    Qbar = self.BubbleFillingFactor(z, zeta)
            #    dndm *= - np.log(1. - Qbar) / Qbar
                
        else:
            raise NotImplementedError('Unrecognized option: %s' % self.pf['bubble_size_dist'])
            
        return Ri, Mi, dndm

    @property
    def halos(self):
        if not hasattr(self, '_halos'):
            self._halos = self.pops[0].halos
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
    
    def BubbleShellRadius(self, z, Ri):
        """
        Given a bubble radius (or array of them), convert to size of
        heated regions.
        """
        
        ##
        # If we made it here, we're doing something fancy.
        ##
        
        if not self.pf['include_temp_fl']:
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
                val = Ri + self.pf['bubble_shell_zsize_zone_{}'.format(ii)]
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
            return self._cache_jp_[z]
        
    def _cache_Vo(self, sep, Ri, Rh, Rc):
        if not hasattr(self, '_cache_Vo_'):
            self._cache_Vo_ = {}
    
        if sep not in self._cache_Vo_:
            self._cache_Vo_[sep] = self.overlap_volumes(sep, Ri, Rh, Rc)

        return self._cache_Vo_[sep]    
    
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
        
        if term[0] != term[1]:
            is_xcorr = True
        else:
            is_xcorr = False
            
        if self.pf['bubble_size_dist'].lower() != 'fzh04':
            raise NotImplemented('Only know fzh04 BSD so far!')

        result = self._cache_jp(z, term)
        if result is not None:
            return result
    
        #if 'c' in term:
        #if term == 'cc' and self.pf['powspec_lya_method'] > 0:                
        #    Rc, Mi, dndm = self.BubblePodSizeDistribution(z, zeta)
        #    Ri = Rh = np.zeros_like(Rc)
        #else:
        Ri, Mi, dndm = self.BubbleSizeDistribution(z, zeta)
        Rh, Rc = self.BubbleShellRadius(z, Ri)
        #Rc = self.BubblePodRadius(z, Ri=Ri, zeta=zeta, 
        #    zeta_lya=zeta_lya)
                        
        # Minimum bubble size            
        Mmin = self.Mmin(z) * zeta
        iM = np.argmin(np.abs(self.halos.M - Mmin))
        iz = np.argmin(np.abs(z - self.halos.z))
        
        # Loop over scales
        B1 = np.zeros(dr.size)
        B2 = np.zeros(dr.size)
        BT = np.zeros(dr.size)
        for i, sep in enumerate(dr):
            # Could do this once externally, i.e., not for each term.
            
            # Yields: V11, V12, V13, V22, V23, V33
            all_V = self._cache_Vo(sep, Ri, Rh, Rc)
        
            # For two-halo terms, need bias of sources.
            if self.pf['include_bias']:
                ep = self.excess_probability(z, sep, data, zeta, zeta_lya, 
                    term)
            else:
                ep = np.zeros_like(self.halos.M)
        
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
                Vo = all_V[0]#self.overlap_region_sphere(sep, Ri)
        
                Vss_ne_1 = 4. * np.pi * Ri**3 / 3. - Vo
                Vss_ne_2 = Vss_ne_1
        
                limiter = 'i'
                
            #elif term == 'in':
            #    Vo = all_V[0]
            #    
            #    _Pin_int = dndm * (4. * np.pi * Ri**3 / 3. - Vo) * Mi
            #    Pin = np.trapz(_Pin_int[iM:], x=np.log(Mi[iM:]))
            #    
            #    Vss_ne_1 = Vss_ne_2 = np.zeros_like(Ri)
            #
            #    limiter = None
                
            elif term == 'id':
                Vo = all_V[0]
                
                delta_B = self._B(z, zeta, zeta)
                _Pin_int = dndm * Vo * Mi * delta_B
                Pin = np.trapz(_Pin_int[iM:], x=np.log(Mi[iM:]))
                
                Vss_ne_1 = Vss_ne_2 = np.zeros_like(Ri)
        
                xi_dd = data['cf_dd'][i]
                bh = self.halos.bias_of_M(z)
                bb = self.bubble_bias(z, zeta)
                
                Vi = 4. * np.pi * Ri**3 / 3.
                
                dndm_h = self.halos.dndm[iz]
                iM_h = np.argmin(np.abs(self.halos.M - self.Mmin(z)))
                
                _prob_i_d = dndm * bb * xi_dd * Vi * Mi
                prob_i_d = np.trapz(_prob_i_d[iM:], x=np.log(Mi[iM:]))
                
                #_Pout = bh * dndm_h
                #Pout = np.trapz(_Pout[iM_h:], x=np.log(self.halos.M[iM_h:]))
                #_Pout_int = 
                #Pout = np.trapz(_Pout_int[iM:], x=np.log(Mi[iM:]))
                Pout = prob_i_d
                
                print z, i, sep, Pin, Pout
                
                limiter = 'i'
        
            elif term == 'hh':
                
                if self.pf['powspec_temp_method'] == 'xset':
                    # In this case, zeta == zeta_X, and Ri is really the radius
                    # of the heated region
                    Vo = all_V[0]
                    
                    Vss_ne_1 = 4. * np.pi * Ri**3 / 3.
                    Vss_ne_2 = Vss_ne_1
                                    
                    limiter = None
                elif self.pf['powspec_temp_method'] == 'shell':    
                
                    Vo = all_V[3]
                    
                    Vss_ne_1 = 4. * np.pi * (Rh - Ri)**3 / 3. \
                             - 0.5 * all_V[1] #(self.IV(sep, Ri, Rc) - self.IV(sep, Ri, Rh))
                    Vss_ne_2 = Vss_ne_1
                                    
                    limiter = None
        
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
                
                xi_dd = data['cf_dd'][i]
                bh = self.halos.bias_of_M(z)
                bb = self.bubble_bias(z, zeta)
                
                Vi = 4. * np.pi * (Rh - Ri)**3 / 3.
                _prob_i_d = dndm * bb * xi_dd * Vi * Mi
                prob_i_d = np.trapz(_prob_i_d[iM:], x=np.log(Mi[iM:]))
                
                Pout = prob_i_d #+ data['Qh']
                
                print z, i, sep, Pin, Pout
                
                limiter = None
                
            else:
                print 'Skipping %s term for now' % term
                #raise NotImplemented('under construction')
                break
        
            # Compute the one-bubble term
            integrand1 = dndm * Vo
        
            exp_int1 = np.exp(-np.trapz(integrand1[iM:] * Mi[iM:],
                x=np.log(Mi[iM:])))
        
            P1 = (1. - exp_int1)
            #P1 = np.log(exp_int1)
            
            Vss_ne_1[Vss_ne_1 < 0] = 0.
            Vss_ne_2[Vss_ne_2 < 0] = 0.
        
            # Start chugging along on two-bubble term                    
            integrand2_1 = dndm * np.maximum(Vss_ne_1, 0.0)
            integrand2_2 = dndm * np.maximum(Vss_ne_2, 0.0)
        
            exp_int2 = np.exp(-np.trapz(integrand2_1[iM:] * Mi[iM:], 
                x=np.log(Mi[iM:])))
            exp_int2_ex = np.exp(-np.trapz(integrand2_2[iM:] * Mi[iM:] \
                * corr[iM:], x=np.log(Mi[iM:])))
                
            ##
            # Second integral sometimes is very nearly (but just in excess)
            # of unity. Don't allow it! Better solution pending...
            ##
            P2 = exp_int1 * (1. - exp_int2) * (1. - exp_int2_ex)
            #P2 = exp_int1 * np.log(exp_int2) * np.log(exp_int2_ex)
        
            B1[i] = P1
            B2[i] = min(max(P2, 0.0), 1.0)
                        
            if limiter is not None:
                Q = data['Q%s' % limiter]
            else:
                Q = 0.
        
            if term in ['id', 'hd']:
                #BT[i] = P1 - Pin
                
                BT[i] = Pin + Pout #- P1

                if limiter is not None:
                    if Q >= 0.5:
                        BT[i] = Pin                

                continue
                    
            # Add optional correction to ensure limiting behavior?
            if (limiter is None) or (not self.pf['powspec_volfix']):
                BT[i] = max(P1 + P2, 0.0)
                continue
        
            if Q < 0.5:
                BT[i] = max(P1 + P2, 0.0)
            else:
                BT[i] = max((1. - Q) * P1 + Q**2, 0.0)
        
        self._cache_jp_[z][term] = BT, B1, B2
        
        return BT, B1, B2


        
