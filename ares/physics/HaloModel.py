# Thanks, Jason Sun, for most of this!

import numpy as np
import scipy.special as sp
from scipy.integrate import quad
from .HaloMassFunction import HaloMassFunction

class HaloModel(HaloMassFunction):
    
    def mvir_to_rvir(self, m):
        return (3. * m / (4. * np.pi * self.pf['halo_delta'] \
            * self.cosm.mean_density0)) ** (1. / 3.)

    def cm_relation(self, m, z, get_rs):
        """
        The concentration-mass relation
        """
        if self.pf['halo_cmr'] == 'duffy':
            return self._cm_duffy(m, z, get_rs)
        elif self.pf['halo_cmr'] == 'zehavi':
            return self._cm_zehavi(m, z, get_rs)
        else:
            raise NotImplemented('help!')

    def _cm_duffy(self, m, z, get_rs=True):
        c = 6.71 * (m / (2e12)) ** -0.091 * (1 + z) ** -0.44
        rvir = self.mvir_to_rvir(m)

        if get_rs:
            return c, rvir / c
        else:
            return c

    def _cm_zehavi(self, m, z, get_rs=True):
        c = ((m / 1.5e13) ** -0.13) * 9.0 / (1 + z)
        rvir = self.mvir_to_rvir(m)

        if get_rs:
            return c, rvir / c
        else:
            return c

    def _dc_nfw(self, c):
        return c** 3. / (4. * np.pi) / (np.log(1 + c) - c / (1 + c))

    def rho_nfw(self, r, m, z):

        c, r_s = self.cm_relation(m, z, get_rs=True)
        
        x = r / r_s
        rn = x / c

        if np.iterable(x):
            result = np.zeros_like(x)
            result[rn <= 1] = (self._dc_nfw(c) / (c * r_s)**3 / (x * (1 + x)**2))[rn <= 1]

            return result
        else:
            if rn <= 1.0:
                return self._dc_nfw(c) / (c * r_s) ** 3 / (x * (1 + x) ** 2)
            else:
                return 0.0

    def u_nfw(self, k, m, z):
        """
        Normalized Fourier Transform of rho.
        
        ..note:: This is Equation 81 from Cooray & Sheth (2002).
        
        Parameters
        ----------
        k : int, float
            Wavenumber
        m : 
        """
        c, r_s = self.cm_relation(m, z, get_rs=True)

        K = k * r_s

        asi, ac = sp.sici((1 + c) * K)
        bs, bc = sp.sici(K)

        # The extra factor of np.log(1 + c) - c / (1 + c)) comes in because
        # there's really a normalization factor of 4 pi rho_s r_s^3 / m, 
        # and m = 4 pi rho_s r_s^3 * the log term
        return (np.sin(K) * (asi - bs) - np.sin(c * K) / ((1 + c) * K) \
            + np.cos(K) * (ac - bc)) / (np.log(1 + c) - c / (1 + c))
        
    def PS_OneHalo(self, z, k, profile_ft=None):
        """
        Compute the one halo term of the halo model for given input profile.
        """
        
        iz = np.argmin(np.abs(z - self.z))
        logMmin = self.logM_min[iz]
        #iM = np.argmin(np.abs(logMmin - self.logM))
        iM = 0
                        
        # Can plug-in any profile, but will default to dark matter halo profile
        if profile_ft is None:
            profile_ft = self.u_nfw

        prof = np.abs(map(lambda M: profile_ft(k, M, z), self.M))
                        
        #if mass_dependence is not None:
        #    prof *= mass_dependence(Mh=self.M, z=z)                
                        
        dndlnm = self.dndm[iz,:] * self.M
        rho_bar = self.mgtm[iz,iM]
                        
        integrand = dndlnm * (self.M / rho_bar)**2 * prof**2
         
        result = np.trapz(integrand[iM:], x=self.lnM[iM:]) 
        
        return result
        
    def PS_TwoHalo(self, z, k, profile_ft=None):
        """
        Compute the two halo term of the halo model for given input profile.
        
        .. note :: Assumption of linearity?
        
        Parameters
        ----------
        
        """
        iz = np.argmin(np.abs(z - self.z))
        logMmin = self.logM_min[iz]
        #iM = np.argmin(np.abs(logMmin - self.logM))
        #iM = 0
        
        # Can plug-in any profile, but will default to dark matter halo profile
        if profile_ft is None:
            profile_ft = self.u_nfw

        prof = np.abs(map(lambda M: profile_ft(k, M, z), self.M))
        
        #if mass_dependence is not None:
        #    Mterm = mass_dependence(Mh=self.M, z=z) 
        #    norm = np.trapz(Mterm, x=self.M)
        #    
        #    prof *= Mterm / norm
                
        # Short-cuts
        dndlnm = self.dndm[iz,:] * self.M
        bias = self.bias_of_M(z)
        rho_bar = self.mgtm[iz,0] # Should be equal to cosmic mean density * fcoll
        
        # Small halo correction.
        # Make use of Cooray & Sheth Eq. 71
        _integrand = dndlnm * (self.M / rho_bar) * bias
        correction = 1. - np.trapz(_integrand, x=self.lnM)
        
        # Compute two-halo integral with profile in there
        integrand = dndlnm * (self.M / rho_bar) * \
            prof * bias
            
        return (np.trapz(integrand, x=self.lnM) + correction)**2 \
            * float(self.psCDM(z, k))

    def PowerSpectrum(self, z, k, profile_ft=None):
        if type(k) == np.ndarray:
            f1 = lambda kk: self.PS_OneHalo(z, kk, profile_ft=profile_ft)
            f2 = lambda kk: self.PS_TwoHalo(z, kk, profile_ft=profile_ft)
            ps1 = np.array(map(f1, k))
            ps2 = np.array(map(f2, k))
            return ps1 + ps2
        else:    
            return self.PS_OneHalo(z, k, profile_ft=profile_ft) \
                 + self.PS_TwoHalo(z, k, profile_ft=profile_ft)
    
    #@property
    #def _tab_ps_dd(self):
    #    if not hasattr(self, '_tab_ps_dd'):
    #        
    #        _z = 'powspec_redshifts'
    #        _r = 'fft_scales'
    #        
    #        self._tab_ps_dd
    
    def table_prefix_hm(self, with_size=False):
        """
        What should we name this table?
        
        Convention:
        ps_FIT_logM_nM_logMmin_logMmax_z_nz_
        
        Read:
        halo mass function using FIT form of the mass function
        using nM mass points between logMmin and logMmax
        using nz redshift points between zmin and zmax
        
        """
        
        M1, M2 = self.pf['hmf_logMmin'], self.pf['hmf_logMmax']
        z1, z2 = self.pf['hmf_zmin'], self.pf['hmf_zmax']
        
        rarr = self.pf['fft_scales']
        
        
        if with_size:
            logMsize = (self.pf['hmf_logMmax'] - self.pf['hmf_logMmin']) \
                / self.pf['hmf_dlogM']                
            zsize = ((self.pf['hmf_zmax'] - self.pf['hmf_zmin']) \
                / self.pf['hmf_dz']) + 1
                
            assert logMsize % 1 == 0
            logMsize = int(logMsize)    
            assert zsize % 1 == 0
            zsize = int(round(zsize, 1))    
                
            return 'ps_%s_logM_%s_%i-%i_z_%s_%i-%i' \
                % (self.hmf_func, logMsize, M1, M2, zsize, z1, z2)
        else:
            return 'ps_%s_logM_*_%i-%i_z_*_%i-%i' \
                % (self.hmf_func, M1, M2, z1, z2)
    
    def tabulate_ps(self, z, k):
        """
        Tabulate the (density) power spectrum.
        """
        
        pb = ProgressBar(len(z), 'ps_dd')
        pb.start()
        
        for i, z in enumerate(self.z):
        
            if i > 0:
                self.MF.update(z=z)
        
            if i % size != rank:
                continue
        
            # Compute collapsed fraction
            #if self.hmf_func == 'PS' and self.hmf_analytic:
            #    delta_c = self.MF.delta_c / self.MF.growth_factor
            #    #fcoll_tab[i] = erfc(delta_c / sqrt2 / self.MF._sigma_0)
            #    
            #else:
        
            # Has units of h**4 / cMpc**3 / Msun
            self.dndm[i] = self.MF.dndm.copy() * self.cosm.h70**4
            self.mgtm[i] = self.MF.rho_gtm.copy() * self.cosm.h70**2
            self.ngtm[i] = self.MF.ngtm.copy() * self.cosm.h70**3
        
            # Remember that mgtm and mean_density have factors of h**2
            # so we're OK here dimensionally
            #fcoll_tab[i] = self.mgtm[i] / self.cosm.mean_density0
        
            # Eq. 3.53 and 3.54 in Steve's book
            #delta_b = 1. #?
            #delta_b0 = delta_b / self.growth_factor
            #nu_c = (self.delta_c - delta_b) / self.sigma
            #delta_c = self.delta_c - delta_b0
        
            delta_sc = (1. + z) * (3. / 5.) * (3. * np.pi / 2.)**(2./3.)
            # Not positive that this shouldn't just be sigma
            nu = (delta_sc / self.MF._sigma_0)**2
        
            # Cooray & Sheth (2002) Equations 68-69
            if self.hmf_func == 'PS':
                self.bias_tab[i] = 1. + (nu - 1.) / delta_sc
        
            elif self.hmf_func == 'ST':
                ap, qp = 0.707, 0.3
        
                self.bias_tab[i] = 1. \
                    + (ap * nu - 1.) / delta_sc \
                    + (2. * qp / delta_sc) / (1. + (ap * nu)**qp)
            else:
                raise NotImplemented('No bias for non-PS non-ST MF yet!')
        
            self.psCDM_tab[i] = self.MF.power / self.cosm.h70**3
        
            self.growth_tab[i] = self.MF.growth_factor            
        
            pb.update(i)
        
        pb.finish()
        
        # All processors will have this.
        self.sigma_tab = self.MF._sigma_0
        
        # Collect results!
        if size > 1:
            #tmp1 = np.zeros_like(fcoll_tab)
            #nothing = MPI.COMM_WORLD.Allreduce(fcoll_tab, tmp1)
            #_fcoll_tab = tmp1
        
            tmp2 = np.zeros_like(self.dndm)
            nothing = MPI.COMM_WORLD.Allreduce(self.dndm, tmp2)
            self.dndm = tmp2
        
            tmp3 = np.zeros_like(self.ngtm)
            nothing = MPI.COMM_WORLD.Allreduce(self.ngtm, tmp3)
            self.ngtm = tmp3
        
            tmp4 = np.zeros_like(self.mgtm)
            nothing = MPI.COMM_WORLD.Allreduce(self.mgtm, tmp4)
            self.mgtm = tmp4
        
            tmp5 = np.zeros_like(self.bias_tab)
            nothing = MPI.COMM_WORLD.Allreduce(self.bias_tab, tmp5)
            self.bias_tab = tmp5
        
            tmp6 = np.zeros_like(self.psCDM_tab)
            nothing = MPI.COMM_WORLD.Allreduce(self.psCDM_tab, tmp6)
            self.psCDM_tab = tmp6
        
            tmp7 = np.zeros_like(self.growth_tab)
            nothing = MPI.COMM_WORLD.Allreduce(self.growth_tab, tmp7)
            self.growth_tab = tmp7
        
        #else:
        #    _fcoll_tab = fcoll_tab   
        
        # Fix NaN elements
        #_fcoll_tab[np.isnan(_fcoll_tab)] = 0.0
        #self._fcoll_tab = _fcoll_tab    