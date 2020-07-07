"""

ExcursionSet.py

Author: Jordan Mirocha
Affiliation: McGill
Created on: Mon 18 Feb 2019 10:38:06 EST

Description: 

"""

import numpy as np
from .Constants import rho_cgs
from .Cosmology import Cosmology
from ..util.Math import central_difference
from ..util.ParameterFile import ParameterFile
from scipy.integrate import simps, quad
from scipy.interpolate import interp1d
from scipy.misc import derivative

two_pi = 2. * np.pi
four_pi = 4. * np.pi
two_pi_sq = 2. * np.pi**2

class ExcursionSet(object):
    def __init__(self, cosm=None, **kwargs):
        self.pf = ParameterFile(**kwargs)
        
        if cosm is not None:
            self._cosm = cosm
       
    @property
    def cosm(self):
        if not hasattr(self, '_cosm'):
            self._cosm = Cosmology(pf=self.pf, **self.pf)
        return self._cosm
        
    @cosm.setter
    def cosm(self, value):
        self._cosm = value
        
    @property
    def tab_sigma(self):
        if not hasattr(self, '_tab_sigma'):
            raise AttributeError('must set by hand for now')
        return self._tab_sigma
        
    @tab_sigma.setter
    def tab_sigma(self, value):
        self._tab_sigma = value
    
    @property
    def tab_M(self):
        if not hasattr(self, '_tab_M'):
            raise AttributeError('must set by hand for now')
        return self._tab_M  
    
    @tab_M.setter
    def tab_M(self, value):
        self._tab_M = value
    
    @property
    def tab_z(self):
        if not hasattr(self, '_tab_z'):
            raise AttributeError('must set by hand for now')
        return self._tab_z
    
    @tab_z.setter
    def tab_z(self, value):
        self._tab_z = value    
    
    @property
    def tab_k(self):
        if not hasattr(self, '_tab_k'):
            raise AttributeError('must set by hand for now')
        return self._tab_k
    
    @tab_k.setter
    def tab_k(self, value):
        self._tab_k = value
    
    @property
    def tab_ps(self):
        if not hasattr(self, '_tab_ps'):
            raise AttributeError('must set by hand for now')
        return self._tab_ps  
    
    @tab_ps.setter
    def tab_ps(self, value):
        self._tab_ps = value
    
    @property
    def tab_growth(self):
        if not hasattr(self, '_tab_growth'):
            raise AttributeError('must set by hand for now')
        return self._tab_growth
    
    @tab_growth.setter
    def tab_growth(self, value):
        self._tab_growth = value    
                
    def _growth_factor(self, z):
        return np.interp(z, self.tab_z, self.tab_growth,
            left=np.inf, right=np.inf)            
                
    def Mass(self, R):
        return self.cosm.rho_m_z0 * rho_cgs * self.WindowVolume(R)
        
    def PDF(self, delta, R):
        pass
    
    def WindowReal(self, x, R):
        """
        Return real-space window function.
        """
        
        assert type(x) == np.ndarray
        
        if self.pf['xset_window'] == 'tophat-real':
            W = np.zeros_like(x)
            W[x <= R] = 3. / four_pi / R**3
        elif self.pf['xset_window'] == 'tophat-fourier':
            W = (np.sin(x / R) - (x / R) * np.cos(x / R)) \
                / R**3 / two_pi_sq / (x / R)**3
        else:
            raise NotImplemented('help')
            
        return W    
        
    def WindowFourier(self, k, R):
        if self.pf['xset_window'] == 'sharp-fourier':
            W = np.zeros_like(k)
            ok = 1. - k * R >= 0.
            W[ok == 1] = 1.
        elif self.pf['xset_window'] == 'tophat-real':
            W = 3. * (np.sin(k * R) - k * R * np.cos(k * R)) / (k * R)**3
        elif self.pf['xset_window'] == 'tophat-fourier':    
            W = np.zeros_like(k)
            W[k <= 1./R] = 1.
        else:
            raise NotImplemented('help')
    
        return W
        
    def WindowVolume(self, R):
        if self.pf['xset_window'] == 'sharp-fourier':
            # Sleight of hand
            return four_pi * R**3 / 3.
        elif self.pf['xset_window'] == 'tophat-real':
            return four_pi * R**3 / 3.
        elif self.pf['xset_window'] == 'tophat-fourier':
            return four_pi * R**3 / 3.
        else:
            raise NotImplemented('help')    
    
    def Variance(self, z, R):
        """
        Compute the variance in the field on some scale `R`.
        """
        
        iz = np.argmin(np.abs(z - self.tab_z))
        
        # Window function
        W = self.WindowFourier(self.tab_k, R)
        
        # Dimensionless power spectrum
        D = self.tab_k**3 * self.tab_ps[iz,:] / two_pi_sq
                        
        return np.trapz(D * np.abs(W)**2, x=np.log(self.tab_k))
        
    def CollapsedFraction(self):
        pass
        
    def SizeDistribution(self, z, R, dcrit=1.686, dzero=0.0):
        """
        Compute the size distribution of objects.
        
        Parameters
        ----------
        z: int, float
            Redshift of interest.
            
        Returns
        -------
        Tuple containing (in order) the radii, masses, and the
        differential size distribution. Each is an array of length
        self.tab_M, i.e., with elements corresponding to the masses
        used to compute the variance of the density field.
            
        """
        
        # Comoving matter density
        rho0_m = self.cosm.rho_m_z0 * rho_cgs

        M = self.Mass(R)
        S = np.array([self.Variance(z, RR) for RR in R])

        _M, _dlnSdlnM = central_difference(np.log(M[-1::-1]), np.log(S[-1::-1]))
        _M = _M[-1::-1]
        dlnSdlnM = _dlnSdlnM[-1::-1]
        dSdM = dlnSdlnM * (S[1:-1] / M[1:-1])

        dFdM = self.FCD(z, R, dcrit, dzero)[1:-1] * np.abs(dSdM)

        # This is, e.g., Eq. 17 in Zentner (2006) 
        # or Eq. 9.38 in Loeb and Furlanetto (2013)
        dndm = rho0_m * np.abs(dFdM) / M[1:-1]

        return R[1:-1], M[1:-1], dndm
        
    def FCD(self, z, R, dcrit=1.686, dzero=0.0):
        """
        First-crossing distribution function.
        
        i.e., dF/dS where S=sigma^2.
        """
        
        S = np.array([self.Variance(z, RR) for RR in R])
        
        norm = (dcrit - dzero) / np.sqrt(two_pi) / S**1.5
        
        p = norm * np.exp(-(dcrit - dzero)**2 / 2. / S)
        
        return p
        
        
        
        