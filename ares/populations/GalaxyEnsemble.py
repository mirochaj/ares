"""

GalaxyEnsemble.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Wed Jul  6 11:02:24 PDT 2016

Description: 

"""

import numpy as np
from .GalaxyCohort import GalaxyCohort
#from scipy.interpolate import RectBivariateSpline

class GalaxyEnsemble(GalaxyCohort):

    @property
    def use_sfh(self):
        if not hasattr(self, '_use_sfh'):
            if self.pf['pop_tab_sfr'] is not None:
                self._use_sfh = True
            else:
                self._use_sfh = False
        return self._use_sfh
        
    @property
    def use_sfe(self):
        return not self.use_sfh

    @property
    def ztab(self):
        if not hasattr(self, '_ztab'):
            self._ztab = self.pf['pop_tab_z']
        return self._ztab
        
    #@property
    #def common_z_grid(self):
    #    """ Are star-formation histories on the same redshift grid? """
    #    if not hasattr(self, '_common_z_grid'):
    #        self._common_z_grid = 
        
    
    @property
    def Mtab(self):
        if not hasattr(self, '_Mtab'):
            self._Mtab = self.pf['pop_tab_Mh']
        return self._Mtab   
        
    @property
    def sfrtab(self):
        if not hasattr(self, '_sfrtab'):
            self._sfrtab = self.pf['pop_tab_sfr']
        return self._sfrtab     

    @property
    def sfr_tab(self):
        """
        SFR as a function of redshift and halo mass.

            ..note:: Units are Msun/yr.
    
        """
        if not hasattr(self, '_sfr_tab'):
            self._sfr_tab = np.zeros([self.halos.Nz, self.halos.Nm])
            for i, z in enumerate(self.halos.z):

                if self.use_sfe:
                    self._sfr_tab[i] = self.eta[i] * self.MAR(z, self.halos.M) \
                        * self.cosm.fbar_over_fcdm * self.SFE(z, self.halos.M)
                else:
                    pass
    
                mask = self.halos.M >= self.Mmin[i]
                self._sfr_tab[i] *= mask
    
        return self._sfr_tab
    
    @property
    def _sfr_spline(self):
        pass
        
    def SFR(self, z, M):
        """
        Star formation rate at redshift z in a halo of mass M.
    
        ..note:: Units should be solar masses per year at this point.
        """
    
        return self._sfr_spline(z, M)
        
    @property
    def sfrd_tab(self):
        """
        SFRD as a function of redshift.
    
            ..note:: Units are g/s/cm^3 (comoving).
    
        """
        if not hasattr(self, '_sfrd_tab'):
            self._sfrd_tab = np.zeros(self.halos.Nz)
            
            for i, z in enumerate(self.halos.z):
                integrand = self.sfr_tab[i] * self.halos.dndlnm[i] \
                    * self.fduty(z, self.halos.M)
 
                tot = np.trapz(integrand, x=self.halos.lnM)
                cumtot = cumtrapz(integrand, x=self.halos.lnM, initial=0.0)
                
                self._sfrd_tab[i] = tot - \
                    np.interp(np.log(self.Mmin[i]), self.halos.lnM, cumtot)
                
            self._sfrd_tab *= g_per_msun / s_per_yr / cm_per_mpc**3
        
        
    def Mstar(self, z):
        pass
        