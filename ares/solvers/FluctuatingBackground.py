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
    
    def _Vo_sphere_SS(self, dr, R):
        if dr >= (2 * R):
            return 0.0
        else:
            return np.pi * (4. * R + dr) * (2. * R - dr)**2 / 12.
            #return 4. * np.pi * R**3 / 3. - np.pi * dr * (R**2 - dr**2 / 12.)
        
    def _Vo_sphere_DS(self, dr, Ri, Ro):
        if dr >= (Ri + Ro):
            return 0.0         
        elif dr <= (Ro - Ri):
            # This means the points are so close that the overlap region
            # of the outer spheres completely engulfs the inner sphere
            return 4. * np.pi * Ri**3 / 3.
        else:
            return np.pi * (Ri + Ro - dr)**2 \
                * (dr**2 + 2. * dr * (Ri + Ro) - 3. * (Ri - Ro)**2) / 12. / dr
        
    def _Vo_shell(self, dr, Ri, Ro):
        """
        Return the overlap region between two spherical shells.
        
        Parameters
        ----------
        dr : int, float
            Separation
        Ri : int, float
            Inner radius (i.e., ionized bubble radius)
        Ro : int, float 
            Outer radius (i.e., radius of heated region, including ionized zone)
        
        """
        
        # For large enough separations, there can be no overlap
        if dr >= (2. * Ro):
            return 0.0, 0.0, 0.0
        else:

            # Full overlap region of two spheres the size of our 
            # bubble plus its shell. The biggest football shape.
            reg1 = self._Vo_sphere_SS(dr, Ro)
            
            # The overlap region between spheres of radius Ri and Ro. 
            # We need to subtract off twice this area (minus a slight 
            # correction in some cases, see below), since sources in this
            # region will ionize one of the points
            reg2 = self._Vo_sphere_DS(dr, Ri, Ro)
            
            # Avoid double-counting for closely separated points
            reg3 = self._Vo_sphere_SS(dr, Ri)

            # We return in three pieces because we can use the second part
            # for the ionization/contrast cross correlation terms.
            return reg1, reg2, reg3
    
    def _Vo_shell_x2(self, dr, Ri, Rm, Ro):
        """
        Return two overlap regions:
            (i) region in which a single source would cause two points
            separated by 'dr' to sit in different shells.
            (ii) region in which a single source would cause two points
            separated by 'dr' to sit one in the outer shell, one in the
            inner sphere.
            
        I should generalize this to return overlap regions in which a source
        is positioned so as to (i) put both points in the core, 
        (ii) both points in the middle shell, (iii) both points in the outer
        shell, etc.s
    
        Parameters
        ----------
        dr : int, float
            Separation
        Ri : int, float
            Inner radius (i.e., ionized bubble radius)
        Ro : int, float 
            Outer radius (i.e., radius of heated region, including ionized zone)
    
        """
    
        # For large enough separations there is no overlap
        if dr >= (2. * Ro):
            return 0.0, 0.0
        else:
            
            dRi = Rm - Ri
            dRo = Ro - Rm
            dRt = Ro - Ri
    
            # Full overlap region of two spheres the size of our 
            # bubble plus its shell. The biggest football shape.
            # This must be > 0 otherwise the 'if' block would've been taken
            reg_oo = self._Vo_sphere_SS(dr, Ro)
            
            # Next three overlap regions zooming in. This one must come next.
            reg_mo = self._Vo_sphere_DS(dr, Rm, Ro)
            
            if reg_mo == 0:
                return 0.0, 0.0
            
            # Not obvious which of these will come next
            reg_io = self._Vo_sphere_DS(dr, Ri, Ro)
            reg_mm = self._Vo_sphere_DS(dr, Rm, Rm)
            
            # Only overlap between outer two shells
            if reg_io == reg_mm == 0:
                return reg_mo, 0.0
            
            if reg_mm > 0 and reg_io == 0:
                return reg_oo - reg_mm, 0.0
            if reg_io > 0 and reg_mm == 0:
                return 2. * reg_io, 0.0
            
            reg_im = self._Vo_sphere_DS(dr, Ri, Rm)   # must be first  
            reg_ii = self._Vo_sphere_SS(dr, Ri)
            
            # Next up: overlap between middle and inner shells.
            # Now we have to worry about stuff like whether the ionized region 
            # fits entirely within a shell
            if reg_im > 0 and reg_ii == 0:
                cap_n_laces_i = reg_mm - 2. * reg_im
                cap_o = reg_oo - 2. * reg_mo + reg_mm
                return reg_oo - 2 * reg_io - cap_n_laces_i - cap_o, \
                    2. * (reg_io - reg_im)
                
            #elif reg_ii > 0:
            #    cap_n_laces_i = reg_mm - 2. * reg_im
            #    cap_o = reg_oo - 2. * reg_mo + reg_mm
            #    return reg_oo - 2 * reg_io - cap_n_laces_i - cap_o, \
            #        2. * (reg_io - reg_im)    
                
            # Last option: innermost regions are overlapping.    
                
            # Want overlap volumes where:
            # (i) same source heats one point and lya-couples the other
            # (ii) same source lya-couples one point and ionizes the other
            
            return 0.0, 0.0

    def overlap_region_sphere(self, dr, R):
        if not hasattr(self, '_overlap_region_sphere'):
            self._overlap_region_sphere = np.vectorize(self._Vo_sphere_SS)
        return self._overlap_region_sphere(dr, R)

    def overlap_region_shell(self, dr, Ri, Ro):
        if not hasattr(self, '_overlap_region_shell'):
            self._overlap_region_shell = np.vectorize(self._Vo_shell)
        return self._overlap_region_shell(dr, Ri, Ro)
    
    def overlap_region_shell_x2(self, dr, Ri, Rm, Ro):
        if not hasattr(self, '_overlap_region_shell_x2'):
            self._overlap_region_shell_x2 = np.vectorize(self._Vo_shell_x2)
        return self._overlap_region_shell_x2(dr, Ri, Rm, Ro)
            
    def BubbleShellFillingFactor(self, z, zeta):
        if self.pf['bubble_size_dist'] is None:
            R_b = self.pf['bubble_size']
            V_b = 4. * np.pi * R_b**3 / 3.
            n_b = self.BubbleDensity(z)
        
            return 1. - np.exp(-n_b * V_b)
        elif self.pf['bubble_size_dist'].lower() == 'fzh04':
            Rb, Mb, dndm = self.BubbleSizeDistribution(z, zeta)
                
            Rs = self.BubbleShellRadius(z, Rb)    
                
            Vsh = 4. * np.pi * (Rs - Rb)**3 / 3.
                    
            Qhot = np.trapz(dndm * Vsh * Mb, x=np.log(Mb))
        
            return Qhot
        else:
            raise NotImplemented('Uncrecognized option for BSD.')

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
        
        if lya and self.pf['bubble_pod_size_func'] in [None, 'const', 'linear']:
            Rc = self.BubblePodRadius(z, Ri, zeta, zeta_lya)
            Vc = 4. * np.pi * Rc**3 / 3.
            
            if self.pf['powspec_rescale_Qlya']:
                Qc = min(zeta_lya * self.halos.fcoll_2d(z, np.log10(self.Mmin(z))), 1)            
            else:
                Qc = np.trapz(dndlnm[iM:] * Vc[iM:], x=np.log(Mi[iM:])) 
                        
            print 'z=%g, Qc=%g' % (z, Qc)
                        
            return min(Qc, 1.0)
            
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
        
    def mean_bubble_bias(self, z, zeta):
        Rb, Mb, dndm = self.BubbleSizeDistribution(z, zeta)
        Vb = 4. * np.pi * Rb**3 / 3.
        
        Mmin = self.Mmin(z) * zeta
        iM = np.argmin(np.abs(Mmin - self.halos.M))
        bHII = self.bubble_bias(z, zeta)
        
        return np.trapz(dndm[iM:] * Vb[iM:] * bHII[iM:] * Mb[iM:], 
            x=np.log(Mb[iM:]))

    def bubble_bias(self, z, zeta):
        iz = np.argmin(np.abs(z - self.halos.z))
        s = self.halos.sigma_0 #* self.halos.growth_factor[iz]

        return 1. + ((self._B(z, zeta, zeta)**2 / s**2 - (1. / self._B0(z, zeta))) \
            / self.halos.growth_factor[iz])
        
    def _B0(self, z, zeta=40.):

        iz = np.argmin(np.abs(z - self.halos.z))
        s = self.halos.sigma_0 #* self.halos.growth_factor[iz]

        Mmin = self.Mmin(z)
        
        # Variance on scale of smallest collapsed object
        sigma_min = np.interp(Mmin, self.halos.M, s)
        return self._delta_c(z) - np.sqrt(2.) * self._K(zeta) * sigma_min
    
    def _B1(self, z, zeta=40.):
        iz = np.argmin(np.abs(z - self.halos.z))
        s = self.halos.sigma_0 #* self.halos.growth_factor[iz]
        
        Mmin = self.Mmin(z)
        sigma_min = np.interp(Mmin, self.halos.M, s)
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
        sigma_min = np.interp(Mmin, self.halos.M, s)
        
        return self._delta_c(z) - np.sqrt(2.) * self._K(zeta) \
            * np.sqrt(sigma_min**2 - s**2)
        
    def BubbleSizeDistribution(self, z, zeta, zeta_lya=None, lya=False):

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

            Ri = ((Mi / rho0) * 0.75 / np.pi)**(1./3.)

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

        #if lya and self.pf['powspec_lya_method'] == 0:
        #    Mc = Mi * (zeta_lya / zeta)
        #    Rc = ((Mc / self.cosm.mean_density0) * 0.75 / np.pi)**(1./3.)
        #
        #    return Rc, Mc, dndm
        #elif lya and self.pf['powspec_lya_method'] == 1:
        #    Rc = self.BubblePodRadius(z, Ri, zeta)
        #    Mc = (4. * np.pi * Rc**3 / 3.) * self.cosm.mean_density0
        #    return Rc, Mc, dndm
        #elif lya and self.pf['powspec_lya_method'] == 2:
        #    Rc, Mc, dndm = self.BubbleSizeDistribution(z, zeta_lya, zeta_lya=None, lya=False)
        #    return Rc, Mc, dndm
        #elif lya:
        #    raise NotImplemented('help please')
            
        #if (z, lya) not in self._bsd_cache:
        #    self._bsd_cache[(z,lya)] = Ri, Mi, dndm
        
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
            return np.zeros_like(Ri)

        if self.pf['bubble_shell_size_func'] == 'mfp':

            sigma = PhotoIonizationCrossSection(self.Emin_X, species=0)
            lmfp_p = 1. / self.cosm.nH(z) / sigma / cm_per_mpc
            lmfp_c = lmfp_p * (1. + z)
            
            # Return value expected to be shell *radius*, not thickness, so
            # add bubble size back in.
            return Ri + lmfp_c
            
        elif self.pf['bubble_shell_size_func'] == 'const':
            Nlyc = 4. * np.pi * (Ri * cm_per_mpc / (1. + z))**3 * self.cosm.nH(z) / 3.
            NX   = 1e-4 * Nlyc
            
            nrg_flux = NX * self.Emin_X * erg_per_ev
            Jx = nrg_flux * 0.2 
            
            #Rh = Ri \
            #   + (3 * Jx / self.cosm.adiabatic_cooling_rate(z) / 4. / np.pi)**(1./3.) \
            #   * (1. + z) / cm_per_mpc

            return Rh

        #else:
        #    raise NotImplemented('help')

        # Just scale the heated region relative to the ionized region
        # in an even simpler way.
        if self.pf['bubble_shell_size_rel'] is not None:
            return Ri * (1. + self.pf['bubble_shell_size_rel'])
        elif self.pf['bubble_shell_size_abs'] is not None:
            return Ri + self.pf['bubble_shell_size_abs']
        elif self.pf['bubble_shell_size_func'] is None:
            return None

    def BubblePodRadius(self, z, Ri, zeta=40., zeta_lya=1.):
        """
        Calling this a 'pod' since it may one day contain multiple bubbles.
        """
        
        if not self.pf['include_lya_fl']:
            return np.zeros_like(Ri)
        
        # This is basically assuming self-similarity
        if self.pf['bubble_pod_size_func'] is None:
            if self.pf['bubble_pod_size_rel'] is not None:
                return Ri * (1. + self.pf['bubble_pod_size_rel'])
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
            
        if self.pf['bubble_size_dist'].lower() == 'fzh04':
            
            Ri, Mi, dndm = self.BubbleSizeDistribution(z, zeta,
                zeta_lya, lya=False)

            if ('h' in term) or ('c' in term):
                Rh = self.BubbleShellRadius(z, Ri)

            if 'c' in term:
                Rc = self.BubblePodRadius(z, Ri=Ri, zeta=zeta, 
                    zeta_lya=zeta_lya)

            # Minimum bubble size            
            Mmin = self.Mmin(z) * zeta
            iM = np.argmin(np.abs(self.halos.M - Mmin))

            # Loop over scales
            AA = np.zeros_like(dr)
            for i, sep in enumerate(dr):
                            
                # For two-halo terms, need bias of sources.
                if self.pf['include_bias']:
                    ep = self.excess_probability(z, sep, data, zeta)
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
                    Vo = self.overlap_region_sphere(sep, Ri)
                    Vss_ne = 4. * np.pi * Ri**3 / 3. - Vo
                    
                    limiter = 'i'
                
                elif term == 'hh':
                    Vo_sh_r1, Vo_sh_r2, Vo_sh_r3 = \
                        self.overlap_region_shell(sep, Ri, Rh)
                        
                    # Region 1 is the full overlap region between two spheres
                    # of radius Rh, and region 2 is the region in which a 
                    # single source would ionize one of the points, so we 
                    # need to subtract it off.
                    Vo = Vo_sh_r1 - 2. * Vo_sh_r2 + Vo_sh_r3

                    Vss_ne = 4. * np.pi * (Rh - Ri)**3 / 3. - Vo
                    
                    limiter = 'h'

                elif term == 'cc':
                    if self.pf['bubble_pod_size_func'] in [None, 'const', 'linear']:
                        Vo_sh_r1, Vo_sh_r2, Vo_sh_r3 = \
                            self.overlap_region_shell(sep, np.maximum(Ri, Rh), Rc)
                        
                        Vo = Vo_sh_r1 - 2. * Vo_sh_r2 + Vo_sh_r3
                        
                        Vss_ne = 4. * np.pi * (Rc - np.maximum(Ri, Rh))**3 / 3. - Vo
                    elif self.pf['bubble_pod_size_func'] == 'approx_sfh':
                        raise NotImplemented('sorry!')
                    elif self.pf['bubble_pod_size_func'] == 'fzh04':
                        Vo = self.overlap_region_sphere(sep, Rc)
                        Vss_ne = 4. * np.pi * Rc**3 / 3. - Vo
                    else:
                        raise NotImplemented('sorry!')

                    limiter = 'c'
                    
                elif term == 'id':
                    Vo = self.overlap_region_sphere(sep, Ri)
                    
                    delta_B = self._B(z, zeta, zeta)
                    _Pin_int = dndm * Vo * Mi * (1. + delta_B)
                    Pin = np.trapz(_Pin_int[iM:], x=np.log(Mi[iM:]))
                    
                    Vss_ne = 0.0

                    #xi_dd = data['xi_dd_c'][i]
                    #
                    #bHII = self.bubble_bias(z, zeta)

                elif term == 'hc':
                    r1, r2 = self.overlap_region_shell_x2(sep, Ri, Rh, Rc)
                    Vo = r1

                    # One point in cold shell of one bubble, another in
                    # heated shell of completely separate bubble.
                    # Need to be a little careful here!
                    Vss_ne = 4. * np.pi * (Rc - Rh)**3 / 3. - Vo
                    
                    # Get rid of volume of cold region around second
                    # bubble, replace with excess heated volume
                    _corr = 4. * np.pi * (Rh - Ri)**3 / 3. - Vo
                    corr *= _corr / (Vss_ne - Vo)
                    
                    limiter = None
                    
                #elif term == 'ih':
                #    Vo_sh_r1, Vo_sh_r2, Vo_sh_r3 = \
                #        self.overlap_region_shell(sep, Ri, Rh)
                #    Vo = 2. * Vo_sh_r2 - Vo_sh_r3
                #                        
                #    Vss_ne = Vo
                #    #Vo_tot = self.overlap_region_sphere(sep, Rh)
                #    #P1 = np.trapz(dndm[iM:] * Vo_ih[iM:] * Mb[iM:], 
                #    #    x=np.log(Mb[iM:]))
                #    #
                #    #AA[i] = P1
                #    #integrand2 = 0.0    
                #    
                #    limiter = None
                
                elif term == 'hc':
                    # This is tricky!
                    # reg_ii, reg_mm, reg_oo, reg_im, reg_io, reg_mo
                    r1, r2 = self.overlap_region_shell_x2(sep, Ri, Rh, Rc)
                    
                    Vo = r1
                    Vss_ne = 0.0
                    
                    limiter = None
                 
                elif term == 'ih':
                    Vo_sh_r1, Vo_sh_r2, Vo_sh_r3 = \
                        self.overlap_region_shell(sep, Ri, Rh)
                    Vo = 2. * Vo_sh_r2 - Vo_sh_r3
                                        
                    Vss_ne = self.overlap_region_sphere(sep, Rh) - Vo
                    
                    limiter = None
                    
                #elif term == 'ic':
                #    
                #    # May get rid of this once general case works
                #    if not self.pf['include_temp_fl']:
                #        Vo_sh_r1, Vo_sh_r2, Vo_sh_r3 = \
                #            self.overlap_region_shell(sep, Rb, Rc)
                #        Vo_ic = 2. * Vo_sh_r2 - Vo_sh_r3
                #
                #        Vo_tot = self.overlap_region_sphere(sep, Rc)
                #        P1 = np.trapz(dndm[iM:] * Vo_ic[iM:] * Mb[iM:], 
                #            x=np.log(Mb[iM:]))
                #
                #        AA[i] = P1
                #        continue
                    
                else:
                    print 'Skipping %s term for now' % term
                    #raise NotImplemented('under construction')
                    break
                    
                # Compute the one bubble term
                integrand1 = dndm * Vo
                
                exp_int1 = np.exp(-np.trapz(integrand1[iM:] * Mi[iM:], 
                    x=np.log(Mi[iM:])))
                
                P1 = (1. - exp_int1)

                # Start chugging along on two-halo term                    
                integrand2 = dndm * Vss_ne

                exp_int2 = np.exp(-np.trapz(integrand2[iM:] * Mi[iM:], 
                    x=np.log(Mi[iM:])))
                exp_int2_ex = np.exp(-np.trapz(integrand2[iM:] * Mi[iM:] \
                    * corr[iM:], x=np.log(Mi[iM:])))

                ##
                # Second integral sometimes is very nearly (but just in excess)
                # of unity. Don't allow it! Better solution pending...
                P2 = exp_int1 * (1. - exp_int2) * min(1. - exp_int2_ex, 0.0)

                #if np.isnan(P1):
                #    print 'hey P1', term, z, i
                #if np.isnan(P2):
                #    print 'hey P2', term, z, i    
                
                #if P2 < 0:
                #    print term, z, i, exp_int2 > 1, exp_int2_ex > 1, exp_int2_ex
                
                if term == 'id':
                    AA[i] = P1 - Pin
                    continue
                
                # Add optional correction to ensure limiting behavior?
                if limiter is None:
                    AA[i] = P1 + P2
                    continue
                
                Q = data['Q%s' % limiter]
                if Q < 0.5:
                    AA[i] = P1 + P2
                else:
                    AA[i] = (1. - Q) * P1 + Q**2
                
                continue
                
                ##
                # Deprecating below
                ##
                
                # Probability that two points are both ionized
                if term == 'ii':
                    integrand1 = dndm[iM:] * Vo_sph[iM:]
                    
                    exp_int1 = np.exp(-np.trapz(integrand1 * Mb[iM:], 
                        x=np.log(Mb[iM:])))

                    # One halo term
                    P1 = (1. - exp_int1)
                    
                    if data['Qi'] > 0.5:
                        AA[i] += (1. - data['Qi']) * P1 + data['Qi']**2
                    else:
                        integrand2 = dndm[iM:] * (Vb[iM:] - Vo_sph[iM:])
                    
                        exp_int2 = np.exp(-np.trapz(integrand2 * Mb[iM:], 
                            x=np.log(Mb[iM:])))
                        
                        exp_int2_ex = np.exp(-np.trapz(integrand2 * Mb[iM:] * (1. + ep), 
                            x=np.log(Mb[iM:])))

                        P2 = exp_int1 * (1. - exp_int2) * (1. - exp_int2_ex)

                        AA[i] += P1 + P2

                elif term == 'id':

                    P1 = np.trapz(dndm[iM:] * Vo_sph[iM:] * Mb[iM:], 
                        x=np.log(Mb[iM:]))

                    #P1_delta = 
                    
                    #integrand1 = dndm[iM:] * Vo_sph[iM:]
                    #
                    #exp_int1 = np.exp(-np.trapz(integrand1 * Mb[iM:], 
                    #    x=np.log(Mb[iM:])))
                    #
                    ## One halo term
                    #P1 = (1. - exp_int1)
                             
                    delta_B = self._B(z, zeta, zeta)[iM:]
                    _Pin_int = dndm[iM:] * Vo_sph[iM:] * Mb[iM:] \
                        * (1. + delta_B)                    
                    Pin = np.trapz(_Pin_int, x=np.log(Mb[iM:]))
                
                    iz = np.argmin(np.abs(z - self.halos.z))
                    #b = self.halos.bias_tab[iz]

                    xi_dd = data['xi_dd_c'][i]

                    bHII = self.bubble_bias(z, zeta)
                    Pout = data['Qi'] - np.trapz(dndm[iM:] * Vo_sph[iM:], x=Mb[iM:]) \
                         + np.trapz(dndm[iM:] * xi_dd * bHII[iM:], x=Mb[iM:])

                    #AA[i] = P1
                    if data['Qi'] <= 0.5:
                        AA[i] = Pin - P1
                        # Really just Pii times mean density of bubble stuff
                    else:
                        AA[i] += Pin + Pout - data['Qi']        
                        
                elif term == 'hh':
                    Vo_sh_r1, Vo_sh_r2, Vo_sh_r3 = \
                        self.overlap_region_shell(sep, Rb, Rh)
                    # Region 1 is the full overlap region between two spheres
                    # of radius Rh, and region 2 is the region in which a 
                    # single source would ionize one of the points, so we 
                    # need to subtract it off.
                    Vo_hh = Vo_sh_r1 - 2. * Vo_sh_r2 + Vo_sh_r3
                    integrand1 = dndm[iM:] * Vo_hh[iM:]
                    exp_int1 = np.exp(-np.trapz(integrand1 * Mb[iM:], 
                        x=np.log(Mb[iM:])))
                    P1 = 1. - exp_int1
                    
                    AA[i] = max(P1, 0)
                    
                    not_Vo = np.maximum(Vh - Vo_hh, 0.0)
                    integrand2 = dndm * not_Vo
                
                    exp_int2 = np.exp(-np.trapz(integrand2 * Mb[iM:], 
                        x=np.log(Mb[iM:])))

                    exp_int2_ex = np.exp(-np.trapz(integrand2 * Mb[iM:] * (1. + ep), 
                        x=np.log(Mb[iM:])))

                    P2 = exp_int1 * (1. - exp_int2) * (1. - exp_int2_ex)

                    AA[i] += max(P2, 0)
                    
                elif term == 'cc':
                    Vo_sh_r1, Vo_sh_r2, Vo_sh_r3 = \
                        self.overlap_region_shell(sep, np.maximum(Rb, Rh), Rc)

                    Vo_cc = Vo_sh_r1 - 2. * Vo_sh_r2 + Vo_sh_r3
                    integrand1 = dndm_c * Vo_cc

                    exp_int1 = np.exp(-np.trapz(integrand1[iM:] * Mc[iM:],
                        x=np.log(Mc[iM:])))
                    P1 = 1. - exp_int1

                    AA[i] = max(P1, 0)

                    #print z, i, sep, exp_int1#, P1, Vo_tot[iM], Vo_tot[iM]

                    # This is the two bubble term
                    not_Vo = np.maximum(Vc - Vo_cc, 0.0)
                    integrand2 = dndm_c * not_Vo
                
                    exp_int2 = np.exp(-np.trapz(integrand2[iM:] * Mc[iM:],
                        x=np.log(Mc[iM:])))

                    exp_int2_ex = np.exp(-np.trapz(integrand2[iM:] * Mc[iM:] * (1. + ep), 
                        x=np.log(Mc[iM:])))

                    P2 = exp_int1 * (1. - exp_int2) * (1. - exp_int2_ex)

                    print z, i, P1, P2

                    AA[i] += max(P2, 0)

                elif term == 'hc':
                    # This is tricky!
                    # reg_ii, reg_mm, reg_oo, reg_im, reg_io, reg_mo
                    r1, r2 = self.overlap_region_shell_x2(sep, Rb, Rh, Rc)
                    
                    P1 = np.trapz(dndm[iM:] * r1[iM:] * Mb[iM:],
                        x=np.log(Mb[iM:]))
                    
                    AA[i] = P1
                 
                elif term == 'ih':
                    Vo_sh_r1, Vo_sh_r2, Vo_sh_r3 = \
                        self.overlap_region_shell(sep, Rb, Rh)
                    Vo_ih = 2. * Vo_sh_r2 - Vo_sh_r3
                                        
                    Vo_tot = self.overlap_region_sphere(sep, Rh)
                    P1 = np.trapz(dndm[iM:] * Vo_ih[iM:] * Mb[iM:], 
                        x=np.log(Mb[iM:]))
                    
                    AA[i] = P1
                    integrand2 = 0.0
                    
                elif term == 'ic':
                    
                    # May get rid of this once general case works
                    if not self.pf['include_temp_fl']:
                        Vo_sh_r1, Vo_sh_r2, Vo_sh_r3 = \
                            self.overlap_region_shell(sep, Rb, Rc)
                        Vo_ic = 2. * Vo_sh_r2 - Vo_sh_r3

                        Vo_tot = self.overlap_region_sphere(sep, Rc)
                        P1 = np.trapz(dndm[iM:] * Vo_ic[iM:] * Mb[iM:], 
                            x=np.log(Mb[iM:]))

                        AA[i] = P1
                        continue
                        
                    ##
                    # General case
                    ##    
                    r1, r2 = self.overlap_region_shell_x2(sep, Rb, Rh, Rc)
                    
                    P1 = np.trapz(dndm[iM:] * r2[iM:] * Mb[iM:],
                        x=np.log(Mb[iM:]))
                    
                    AA[i] = P1
                    
                    
                    #Vo_sh_r1, Vo_sh_r2 = self.overlap_region_shell(sep, Rb, Ra-Rb)
                    #Vo_ai = 2 * Vo_sh_r2
                    #Vo_tot = self.overlap_region_sphere(sep, Ra)
                    #integrand1 = dndm[iM:] * Vo_ai[iM:]
                    #integrand2 = 0.0    
                elif term == 'dh':
                    pass
                elif term == 'dc':
                    pass    
                elif term == 'xdco':
                    pass    
                    
                else:
                    raise NotImplementedError('help!')

                ##
                # This stuff: only for auto-correlations?
                ##

                

                    #integrand2 *= (1. + ep[iM:])

                #exp_int2 = np.exp(-np.trapz(integrand2 * Mb[iM:], 
                #    x=np.log(Mb[iM:])))
                #
                #AA[i] += exp_int1 * (1. - exp_int2)**2

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
    
    def excess_probability(self, z, r, data, zeta):
        """
        This is the excess probability that a point is ionized given that 
        we already know another point (at distance r) is ionized.
        """
        
        bHII = self.bubble_bias(z, zeta)
        bbar = self.mean_bubble_bias(z, zeta) / data['Qi']

        xi_dd = np.interp(r, data['dr'], data['cf_dd'].real)

        return bHII * bbar * np.array(xi_dd)

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
            
        