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

    def MAB_to_L(self, mag):
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

        # Apparent magnitude
        d10 = 10 * cm_per_pc

        # Luminosity!
        return 10**(mag / -2.5) * flux_AB * 4. * np.pi * d10**2

    def L_to_MAB(self, L):
        d10 = 10 * cm_per_pc
        return -2.5 * np.log10(L / 4. / np.pi / d10**2 / flux_AB)

    def L_to_mab(self, L, z=None, dL=None):
        raise NotImplemented('do we ever use this?')

        # apparent magnitude
        assert (z is not None) or (dL is not None)

        if z is not None:
            dL = self.cosm.LuminosityDistance(z)
        #
        return -2.5 * np.log10(L  / (flux_AB * 4. * np.pi * dL**2))
