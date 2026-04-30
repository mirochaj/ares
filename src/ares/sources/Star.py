"""

StellarSource.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Jul  8 09:56:35 MDT 2013

Description:

"""

import numpy as np
from .Source import Source
from scipy.integrate import quad
from ..util.ParameterFile import ParameterFile
from ..util.SetDefaultParameterValues import StellarParameters
from ..physics.Constants import h, erg_per_ev, k_B, c, sigma_SB, s_per_myr

def _Planck(E, T):
    """ Returns specific intensity of blackbody at temperature T [K]."""

    nu = E * erg_per_ev / h
    return 2.0 * h * nu**3 / c**2 / (np.exp(h * nu / k_B / T) - 1.0)

class Star(Source):
    """ Class for creation and manipulation of stellar radiation sources. """
    def __init__(self, **kwargs):
        """
        Initialize a Star object.

        ..note:: Only support stars with blackbody SEDs.

        Parameters
        ----------
        pf: dict
            Full parameter file.
        src_pars: dict
            Contains source-specific parameters.
        spec_pars: dict
            Contains spectrum-specific parameters.

        Relevant Keyword Arguments
        --------------------------
        source_temperature : int, flot
            Temperature of blackbody to use to create SED.
        source_qdot : int, float
            Number of photons emitted in the range ``(EminNorm, EmaxNorm)``
        source_lifetime : int, float
            Source will switch off after this amount of time [Myr]

        """

        Source.__init__(self, **kwargs)

        self._name = 'star'
        self.Q = self.pf['source_qdot']
        self.T = self.pf['source_temperature']

        # Number of ionizing photons per cm^2 of surface area for BB of
        # temperature self.T. Use to solve for stellar radius (which we need
        # to get Lbol).  The factor of pi gets rid of the / sr units
        QNorm = np.pi * 2. * (k_B * self.T)**3 * \
                quad(lambda x: x**2 / (np.exp(x) - 1.),
                13.6 * erg_per_ev / k_B / self.T, np.inf)[0] \
                / h**3 / c**2

        # "Analytic" solution has poly-logarithm function - not in scipy (yet)
        self.R = np.sqrt(self.Q / 4. / np.pi / QNorm)
        self.Lbol0 = 4. * np.pi * self.R**2 * sigma_SB * self.T**4

        # No time evolution is implicit for such simple spectra
        self.Lbol = lambda t: self.Lbol0
        self.lifetime = self.pf['source_lifetime']

    def Luminosity(self, t=None):
        return self.Lbol

    def _Intensity(self, E, t=None):
        """
        Return quantity *proportional* to fraction of bolometric luminosity.

        Parameters
        ----------
        E : int, float
            Photon energy of interest [eV]

        """

        return _Planck(E, self.T)
