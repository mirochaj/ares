"""

DustExtinction.py

Author: Jordan Mirocha
Affiliation: JPL / Caltech
Created on: Tue Feb 21 13:55:47 PST 2023

Description:

"""

import os
import glob
import numpy as np
from ..data import ARES
from ..util import ParameterFile

try:
    from astropy.io import fits
except ImportError:
    pass

try:
    from dust_extinction.grain_models import WD01
    have_dustext = True
except ImportError:
    have_dustext = False

class DustExtinction(object):
    def __init__(self, **kwargs):
        self.pf = ParameterFile(**kwargs)

    @property
    def method(self):
        return self.pf['pop_dustext_template']

    def get_filename(self):
        """
        Get appropriate filename using regular expression matching.
        """

        prefix = self.pf['pop_dustext_template']
        cands = glob.glob(f'{ARES}/extinction/{prefix}*')

        if len(cands) == 0:
            raise IOError(f'No files found with prefix={prefix}!')
        elif len(cands) > 1:
            raise IOError(f'Multiple files found for dustext_template={prefix}: {cands}')
        else:
            return cands[0]

    @property
    def _dustext_instance(self):
        if not hasattr(self, '_dustext_instance_'):
            assert have_dustext, "Need dust_extinction package for this!"
            mth1, curve = self.method.split(':')

            self._dustext_instance_ = WD01(curve)
            assert mth1 == 'WD01'

        return self._dustext_instance_

    @property
    def tab_x(self):
        """
        1/wavelengths [micron^-1].
        """
        if not hasattr(self, '_tab_x'):
            self._tab_x = 1e4 / self.tab_waves_c
        return self._tab_x

    @property
    def tab_waves_c(self):
        """
        Wavelengths in Angstroms.
        """
        if not hasattr(self, '_tab_waves_c'):
            if self.method.startswith('WD01'):
                self._tab_waves_c = 1e4 / self._dustext_instance.data_x
            else:
                self._load()
        return self._tab_waves_c

    @property
    def tab_extinction(self):
        """
        Lookup table of Rv=Av/E(B-V).
        """
        if not hasattr(self, '_tab_extinction'):
            if self.method.startswith('WD01'):
                # Callable expects [x] = 1 / microns
                from astropy.units import micron
                self._tab_extinction = self._dustext_instance(self.tab_x / micron)
            else:
                raise NotImplemented('issue with E(B-V)-based lookup tab not resolved.')
                self._load()
        return self._tab_extinction

    def _load(self):
        """
        Load appropriate dust extinction lookup table from disk.

        Returns
        -------
        Tuple containing (wavelengths in Angstroms, Av/E(B-V)).
        """

        fn = self.get_filename()

        hdu = fits.open(fn)
        data = hdu[1].data
        invwave = []
        extinct = []
        for element in data:
            invwave.append(element[0])
            extinct.append(element[1])

        invwave = np.array(invwave)
        isort = np.argsort(invwave)
        invwave = invwave[isort]
        extinct = np.array(extinct)[isort]

        if self.method.startswith('xgal'):
            self._tab_waves_c = np.array(invwave)
            self._tab_extinction = np.array(extinct)
        else:
            self._tab_waves_c = 1e4 / np.array(invwave)[-1::-1]
            self._tab_extinction = np.array(extinct)[-1::-1]

    def get_curve(self, wave):
        """
        Get extinction (or attenuation) curve from lookup table.

        .. note :: This is what is contained in attenuation curves natively.

        Parameters
        ----------
        wave : int, float, np.ndarray
            Wavelength [Angstroms]

        """
        return np.interp(wave, self.tab_waves_c, self.tab_extinction)

    def get_A_lam(self, wave, Av=0.1):
        return self.get_curve(wave) * Av

    def get_slope(self, wave1, wave2):
        return self.get_A_lam(wave1) / self.get_A_lam(wave2)

    def get_uv_slope(self):
        return self.get_slope(1500., 3000.)

    def get_optical_slope(self):
        return self.get_slope(4400., 5500.)

    def get_Rv(self):
        return 1. / (self.get_optical_slope() - 1.)

    def get_A2175(self, baseline=False):
        if baseline:
            return 0.33 * self.get_A_lam(1500.) + 0.67 * self.get_A_lam(3000.)
        else:
            return self.get_A_lam(2175)

    def get_B(self):
        """
        Get "UV bump" strength (Salim & Narayanan 2018 Eq. 4)
        """
        return self.get_A2175() / self.get_A2175(baseline=True)

    def get_tau_lam(self, wave, Av=0.1):
        """
        Return optical depth corresponding to given extinction.
        """
        A = self.get_A_lam(wave, Av=Av)
        return np.log(10) * A / 2.5

    def get_T_lam(self, wave, Av=0.1):
        """
        Return transmission (equivalent to e^-tau) corresponding to extinction.
        """
        return 10**(-self.get_A_lam(wave, Av=Av) / 2.5)
