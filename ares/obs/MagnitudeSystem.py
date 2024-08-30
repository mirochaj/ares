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

d10 = 10 * cm_per_pc

class MagnitudeSystem(object):
    def __init__(self, cosm=None, **kwargs):
        if cosm is None:
            self.cosm = Cosmology(**kwargs)
        else:
            self.cosm = cosm

    #def MAB_to_L(self, mag):
    #    """
    #    Convert AB magnitude [ABSOLUTE] to rest-frame luminosity.

    #    Parameters
    #    ----------
    #    mag : int, float
    #        Absolute magnitude in AB system.
    #    z : int, float
    #        Redshift of object
    #    dL : int, float
    #        Luminosity distance of object [cm]

    #    Currently this is dumb: doesn't need to depend on luminosity.

    #    Returns
    #    -------
    #    Luminosity in erg / s / Hz.

    #    """

    #    # Apparent magnitude
    #    d10 = 10 * cm_per_pc

    #    # Luminosity!
    #    return 10**(mag / -2.5) * flux_AB * 4. * np.pi * d10**2

    def get_lum_from_mag_app(self, z, mags):
        mag_abs = self.get_mags_abs(z, mags)

        return 10**(mag_abs / -2.5) * flux_AB * 4. * np.pi * d10**2

    def get_mag_abs_from_lum(self, L):
        return -2.5 * np.log10(L / 4. / np.pi / d10**2 / flux_AB)

    def get_mag_app_from_lum(self, z, L):
        mag_abs = self.get_mag_abs_from_lum(L)
        return get_mags_app(z, mag_abs)

    def get_mags_abs(self, z, mags):
        """
        Convert apparent magnitudes to absolute magnitudes.
        """
        d_pc = self.cosm.LuminosityDistance(z) / cm_per_pc
        return mags - 5 * np.log10(d_pc / 10.) + 2.5 * np.log10(1. + z)

    def get_mags_app(self, z, mags):
        """
        Convert absolute magnitudes to apparent magnitudes.
        """
        d_pc = self.cosm.LuminosityDistance(z) / cm_per_pc
        return mags + 5 * np.log10(d_pc / 10.) - 2.5 * np.log10(1. + z)


    def L_to_MAB(self, L):
        return self.get_mag_abs_from_lum(L)

    def L_to_mab(self, L, z=None, dL=None):
        # apparent magnitude
        assert (z is not None) or (dL is not None)

        if z is not None:
            dL = self.cosm.LuminosityDistance(z)
        #
        return -2.5 * np.log10(L  / (flux_AB * 4. * np.pi * dL**2))
