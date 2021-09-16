"""

OpticalDepth.py

Author: Jordan Mirocha
Affiliation: McGill
Created on: Mon 27 Jan 2020 09:44:27 EST

Description: Analytic fits to IGM optical depth models.

"""

import numpy as np
from ..physics import Hydrogen, Cosmology
from ..util.ParameterFile import ParameterFile
from ..physics.Constants import h_p, c, erg_per_ev, lam_LL, lam_LyA

class Madau1995(object):
    def __init__(self, hydr=None, **kwargs):
        self.pf = ParameterFile(**kwargs)
        self.hydr = hydr

    @property
    def cosm(self):
        if not hasattr(self, '_cosm'):
            self._cosm = Cosmology(pf=self.pf, **self.pf)
        return self._cosm

    @property
    def hydr(self):
        if not hasattr(self, '_hydr'):
            self._hydr = Hydrogen(pf=self.pf, cosm=self.cosm, **self.pf)
        elif self._hydr is None:
            self._hydr = Hydrogen(pf=self.pf, cosm=self.cosm, **self.pf)

        return self._hydr

    @hydr.setter
    def hydr(self, value):
        self._hydr = value

    def __call__(self, z, owaves, l_tol=1e-8):

        """
        Compute optical depth of photons at observed wavelengths `owaves`
        emitted by object(s) at redshift `z`.

        Parameters
        ----------
        z : int, float
            Redshift of interest.
        owaves : np.ndarray
            Observed wavelengths in microns.

        Returns
        -------
        Optical depth at all wavelengths assuming Madau (1995) model.

        """
        rwaves = owaves * 1e4 / (1. + z)
        tau = np.zeros_like(owaves)

        # Text just after Eq. 15.
        A = 0.0036, 1.7e-3, 1.2e-3, 9.3e-4
        l = [h_p * c * 1e8 / (self.hydr.ELyn(n) * erg_per_ev) \
            for n in range(2, 7)]

        for i in range(len(A)):
            ok = np.logical_and(rwaves <= l[i], rwaves > l[i+1])

            # Need to be careful about machine precision issues at line centers
            dl = rwaves - l[i]
            k = np.argmin(np.abs(dl))
            if not ok[k]:
                if np.abs(dl[k]) < l_tol:
                    ok[k] = True

            tau[ok==1] += A[i] * (owaves[ok==1] * 1e4 / l[i])**3.46

        #tau[np.logical_and(rwaves < l[-1], rwaves > 912.)] = np.inf

        # Metals
        tau += 0.0017 * (owaves * 1e4 / l[0])**1.68

        # Photo-electric absorption. This is footnote 3 in Madau (1995).
        xem = 1. + z
        xc  = owaves * 1e4 / l[0]
        tau_bf = 0.25 * xc**3 * (xem**0.46 - xc**0.46) \
               + 9.4 * xc**1.5 * (xem**0.18 - xc**0.18) \
               - 0.7 * xc**3 * (xc**-1.32 - xem**-1.32) \
               - 0.023 * (xem**1.68 - xc**1.68)

        tau[rwaves < lam_LL] += tau_bf[rwaves < lam_LL]

        return tau
