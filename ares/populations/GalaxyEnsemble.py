"""

GalaxyEnsemble.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Wed Jul  6 11:02:24 PDT 2016

Description: 

"""

import numpy as np
from .GalaxyCohort import GalaxyCohort

class GalaxyEnsemble(GalaxyCohort):

    @property
    def use_sfh(self):
        if not hasattr(self, '_use_sfh'):
            if self.pf['pop_sfr'] is not None:
                self._use_sfh = True
            else:
                self._use_sfh = False
        return self._use_sfh
        
    @property
    def use_sfe(self):
        return not self.use_sfh

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
                    raise NotImplemented('help')
    
                mask = self.halos.M >= self.Mmin[i]
                self._sfr_tab[i] *= mask
    
        return self._sfr_tab