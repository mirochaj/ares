"""

Population.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu May 28 13:59:41 MDT 2015

Description: 

"""

import numpy as np
from ..physics import Cosmology
from ..util import ParameterFile
from scipy.integrate import quad
from ..physics.Constants import cm_per_mpc

log10 = np.log(10.)    # for when we integrate in log-space    

class Population(object):
    def __init__(self, grid=None, **kwargs):
        self.pf = ParameterFile(**kwargs)
        self.grid = grid
        
        self.zform = self.pf['pop_zform']
        self.zdead = self.pf['pop_zdead']
                
    @property
    def cosm(self):
        if not hasattr(self, '_cosm'):    
            if self.grid is None:
                self._cosm = Cosmology(
                    omega_m_0=self.pf['omega_m_0'], 
                    omega_l_0=self.pf['omega_l_0'], 
                    omega_b_0=self.pf['omega_b_0'],  
                    hubble_0=self.pf['hubble_0'],  
                    helium_by_number=self.pf['helium_by_number'], 
                    cmb_temp_0=self.pf['cmb_temp_0'], 
                    approx_highz=self.pf['approx_highz'], 
                    sigma_8=self.pf['sigma_8'], 
                    primordial_index=self.pf['primordial_index'])
            else:
                self._cosm = grid.cosm
                
        return self._cosm

    def LuminosityDensity(self, z, Emin=None, Emax=None, Lmin=None, Lmax=None):
        """
        Return the luminosity density in the EminNorm-EmaxNorm band.
        
        Parameters
        ----------
        z : int, flot
            Redshift of interest.
            
        Returns
        -------
        Luminosity density in erg / s / c-cm**3.
            
        """
        
        # This means the luminosity density is determined by the SFRD
        if self.rhoL_from_sfrd:
            return self.Emissivity(z, Emin=Emin, Emax=Emax)
        
        if Lmin is None:
            Lmin = 1e41
        if Lmax is None:
            Lmax = 1e42
        
        integrand = lambda LL: 10**LL * self._lf(10**LL, z=z)
        band_conv = self._convert_band(Emin, Emax)
                        
        mult = band_conv / cm_per_mpc**3                
                        
        return quad(integrand, np.log10(Lmin), np.log10(Lmax))[0] * mult
        
    def SpaceDensity(self, z, Emin=None, Emax=None, Lmin=None, Lmax=None):
        """
        Return the luminosity density in the EminNorm-EmaxNorm band.
    
        Parameters
        ----------
        z : int, flot
            Redshift of interest.
    
        Returns
        -------
        Luminosity density in erg / s / cMpc**3.    
    
        """
    
        # This means the luminosity density is determined by the SFRD
        if self.rhoL_from_sfrd:
            raise NotImplemented('help!')
            return self.Emissivity(z, Emin=Emin, Emax=Emax)
    
        if Lmin is None:
            Lmin = 1e41
        if Lmax is None:
            Lmax = 1e42
    
        integrand = lambda LL: self._lf(10**LL, z=z)
    
        return quad(integrand, np.log10(Lmin), np.log10(Lmax))[0]
    