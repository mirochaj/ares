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
from ..util.Misc import numeric_types
from functools import cached_property
from ..phenom.ParameterizedQuantity import get_function_from_par

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

    @cached_property
    def method(self):
        return self.pf['pop_dust_template']

    @cached_property
    def is_template(self):
        is_templ = self.pf['pop_dust_template'] is not None
        if is_templ:
            assert have_dustext
        return is_templ

    @cached_property
    def is_irxb(self):
        return False
        #raise NotImplemented('help')
        #return self.pf['pop_dustext_template'] is not None

    @cached_property
    def is_parameterized(self):
        return self.pf['pop_dust_absorption_coeff'] is not None

    def get_filename(self):
        """
        Get appropriate filename using regular expression matching.
        """

        assert self.is_template, \
            "Only need filename if we're pulling dust extinction from a template."

        prefix = self.pf['pop_dust_template']
        cands = glob.glob(f'{ARES}/extinction/{prefix}*')

        if len(cands) == 0:
            raise IOError(f'No files found with prefix={prefix}!')
        elif len(cands) > 1:
            raise IOError(f'Multiple files found for dust_template={prefix}: {cands}')
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

    def get_transmission(self, wave, Av=None, Sd=None, MUV=None):
        """
        Return the dust transmission at user-supplied wavelength [Angstroms].

        .. note :: Transmission is equivalent to e^-tau, where tau is the dust
            optical depth.

        .. note :: Only one of the optional keyword arguments will actually be
            used, which one depends on methodology.

        Parameters
        ----------
        wave : int, float
            Wavelength of interest [Angstroms].
        Av : int, float, np.ndarray
            Visual extinction in magnitudes [optional].
        Sd : int, float, np.ndarray
            Dust surface density in g / cm^2 [optional].

        Returns
        -------
        Fraction of intensity transmissted at input wavelength.

        """

        if self.is_template:
            return 10**(-self.get_attenuation(wave, Av=Av) / 2.5)
        elif self.is_parameterized:
            tau = self.get_opacity(wave, Av=Av, Sd=Sd)
            return np.exp(-tau)
        else:
            raise NotImplemented('help')

    def get_curve(self, wave):
        """
        Get extinction (or attenuation) curve from lookup table.

        .. note :: This is what is contained in attenuation curves natively.

        Parameters
        ----------
        wave : int, float, np.ndarray
            Wavelength [Angstroms]

        """

        if self.is_template:
            return np.interp(wave, self.tab_waves_c, self.tab_extinction)
        else:
            raise NotImplemented('help')

    def get_attenuation(self, wave, Av=None, Sd=None):
        if type(wave) in numeric_types:
            wave = np.array([wave])

        if self.is_template:
            if type(Av) in numeric_types:
                Av = np.array([Av])
            A = self.get_curve(wave)[None,:] * Av[:,None]
        elif self.is_parameterized:
            if type(Sd) in numeric_types:
                Sd = np.array([Sd])
            tau = self.get_opacity(wave, Av=Av, Sd=Sd)
            A = 2.5 * tau / np.log(10.)
        else:
            raise NotImplemented('help')

        if type(wave) in numeric_types:
            return A[:,0]
        else:
            return A

    def get_opacity(self, wave, Av=None, Sd=None):
        """
        Compute dust opacity at wavelength `wave`.

        Parameters
        ----------
        wave : int, float, np.ndarray
            Wavelength [Angstroms]

        Returns
        -------
        Opacity (dimensionless) for all halos in population. If input `wave` is
        a scalar, returns an array of length `self.halos.tab_M`. If `wave` is
        an array, the return will be a 2-D array with shape
        (len(Mh), len(waves)).
        """
        if type(wave) in numeric_types:
            wave = np.array([wave])

        if self.is_template:
            if type(Av) in numeric_types:
                Av = np.array([Av])
            # Alternatively: tau = np.log(10) * attenuation / 2.5
            T = self.get_transmission(wave, Av=Av, Sd=Sd)
            tau = -np.log(T)
        elif self.is_parameterized:
            if type(Sd) in numeric_types:
                Sd = np.array([Sd])
            kappa = self.get_absorption_coeff(wave=wave)
            tau = kappa[None,:] * Sd[:,None]
        else:
            raise NotImplemented('help')

        if wave.size == 1:
            return tau[:,0]
        else:
            return tau

    def get_absorption_coeff(self, wave):
        """
        Get dust absorption coefficient [cm^2 / g].
        """

        assert self.is_parameterized

        if not hasattr(self, '_get_kappa_'):
            self._get_kappa_ = get_function_from_par('pop_dust_absorption_coeff',
                self.pf)
        return self._get_kappa_(wave=wave)
