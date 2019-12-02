"""

MagnitudeSystem.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sat Sep  5 15:51:39 PDT 2015

Description: Utilities for converting magnitudes to luminosities and such.

"""

import numpy as np
from ..physics.Cosmology import Cosmology
from ..physics.Constants import cm_per_pc, flux_AB

class MagnitudeSystem(object):
    def __init__(self, cosm=None, **kwargs):
        if cosm is None:
            self.cosm = Cosmology(**kwargs)
        else:
            self.cosm = cosm
    
    def mab_to_L(self, mag, z=None, dL=None):
        """
        Convert AB magnitude [APPARENT] to rest-frame luminosity.
        """
        pass
    
    def MAB_to_L(self, mag, z=None, dL=None):
        """
        Convert AB magnitude [ABSOLUTE] to rest-frame luminosity.
        
        Parameters
        ----------
        mag : int, float
            Absolute magnitude in AB system.
        z : int, float
            Redshift of object
        dL : int, float
            Luminosity distance of object [cm]

        Currently this is dumb: doesn't need to depend on luminosity.

        Returns
        -------
        Luminosity in erg / s / Hz.
            
        """
        
        assert (z is not None) or (dL is not None)
        
        if z is not None:
            dL = self.cosm.LuminosityDistance(z)
        
        # Apparent magnitude
        m = mag + 5. * (np.log10(dL / cm_per_pc) - 1.)

        # Luminosity!    
        return 10**(m / -2.5) * flux_AB * 4. * np.pi * dL**2

    def L_to_MAB(self, L, z=None, dL=None):
        # absolute magnitude
        assert (z is not None) or (dL is not None)
        
        if z is not None:
            dL = self.cosm.LuminosityDistance(z)
         
        #    
        m = -2.5 * np.log10(L  / (flux_AB * 4. * np.pi * dL**2))
        
        return m - 5. * (np.log10(dL / cm_per_pc) - 1.)

    def L_to_mab(self, L, z=None, dL=None):
        # apparent magnitude
        assert (z is not None) or (dL is not None)
        
        if z is not None:
            dL = self.cosm.LuminosityDistance(z)
        #    
        return -2.5 * np.log10(L  / (flux_AB * 4. * np.pi * dL**2))

    def mAB_to_flux_density(self, mag):
        pass
    
