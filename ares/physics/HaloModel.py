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
        c = 6.71 * (m / (2.0 * 10 ** 12)) ** -0.091 * (1 + z) ** -0.44
        rvir = self.mvir_to_rvir(m)

        if get_rs:
            return c, rvir / c
        else:
            return c

    def _cm_zehavi(self, m, z, get_rs=True):
        c = ((m / 1.5E13) ** -0.13) * 9.0 / (1 + z)
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
        """ Normalized Fourier Transform of rho """
        c, r_s = self.cm_relation(m, z, get_rs=True)

        K = k * r_s

        asi, ac = sp.sici((1 + c) * K)
        bs, bc = sp.sici(K)

        return (np.sin(K) * (asi - bs) - np.sin(c * K) / ((1 + c) * K) \
            + np.cos(K) * (ac - bc)) / (np.log(1 + c) - c / (1 + c))
        
    def PS_OneHalo(self, z, k, profile_ft=None, profile=None):
        """
        Compute the one halo term of the halo model for given input profile.
        """
        
        iz = np.argmin(np.abs(z - self.z))
        logMmin = self.logM_min[iz]
        iM = np.argmin(np.abs(logMmin - self.logM))
        
        fcoll = float(self.fcoll_2d(z, logMmin))
        
        # Can plug-in any profile, but will default to dark matter halo profile
        if profile_ft is None:
            profile_ft = self.u_nfw

        prof = np.abs(map(lambda M: profile_ft(k, M, z), self.M))
                        
        #if mass_dependence is not None:
        #    prof *= mass_dependence(Mh=self.M, z=z)                
                        
        dndlnm = self.dndm[iz,:] * self.M
                
        integrand = dndlnm * \
            (self.M / fcoll / self.cosm.mean_density0)**2 * prof**2
         
        result = np.trapz(integrand[iM:], x=self.lnM[iM:]) 
        
        return result
        
    def PS_TwoHalo(self, z, k, profile_ft=None, profile=None):
        """
        Compute the two halo term of the halo model for given input profile.
        
        .. note :: Assumption of linearity?
        
        Parameters
        ----------
        
        """
        iz = np.argmin(np.abs(z - self.z))
        logMmin = self.logM_min[iz]
        iM = np.argmin(np.abs(logMmin - self.logM))
        fcoll = float(self.fcoll_2d(z, logMmin))

        # Can plug-in any profile, but will default to dark matter halo profile
        if profile_ft is None:
            profile_ft = self.u_nfw

        prof = np.abs(map(lambda M: profile_ft(k, M, z), self.M))
        
        #if mass_dependence is not None:
        #    Mterm = mass_dependence(Mh=self.M, z=z) 
        #    norm = np.trapz(Mterm, x=self.M)
        #    
        #    prof *= Mterm / norm
        
        dndlnm = self.dndm[iz,:] * self.M

        integrand = dndlnm * \
            (self.M / fcoll / self.cosm.mean_density0) * \
            prof * self.bias_tab[iz,:]
            
        return np.trapz(integrand[iM:], x=self.lnM[iM:])**2 \
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
    
    def CorrelationFunction(self, z, r, profile_ft=None, profile=None):
        k = 2. * np.pi / r
        return np.fft.ifft(self.PowerSpectrum(z, k, profile_ft=profile_ft))
    
        
        