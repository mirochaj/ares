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
from types import FunctionType
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

try:
    from dust_attenuation.averages import C00
    have_dustatt = True
except ImportError:
    have_dustatt = False

# These are AUV = a + b * beta, with a and b in that order in these tuples
_coeff_irxb = {
    'meurer1999': (4.43, 1.99),
    'pettini1998': (1.49, 0.68),
    'capak2015': (0.312, 0.176),
}

# These are beta = a + b * (mag + 19.5)
_coeff_b14 = {
 'lowz': (-1.7, -0.2),
 4: (-1.85, -0.11),
 5: (-1.91, -0.14),
 6: (-2.00, -0.20),
 7: (-2.05, -0.20),
 8: (-2.13, -0.20),
 'highz': (-2, -0.15),
}

class DustExtinction(object):
    def __init__(self, pf=None, **kwargs):
        if pf is None:
            self.pf = ParameterFile(**kwargs)
        else:
            self.pf = pf

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
        return (self.pf['pop_muvbeta'] is not None) and \
            (self.pf['pop_irxbeta'] is not None)

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
    def _dustatt_instance(self):
        if not hasattr(self, '_dustatt_instance_'):
            assert have_dustatt, "Need dust_attenuation package for this!"
            self._dustatt_instance_ = C00
            assert self.method == 'C00'

        return self._dustatt_instance_

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
            elif self.method.startswith('C00'):
                # Just to grab wavelength range
                CC = self._dustatt_instance(Av=1)
                self._tab_waves_c = np.arange(1e4 * CC.x_range[0],
                    1e4 * CC.x_range[1])
            else:
                self._load()
        return self._tab_waves_c

    @property
    def tab_extinction(self):
        """
        Lookup table of Rv=Av/E(B-V).
        """
        if not hasattr(self, '_tab_extinction'):
            # Callable expects [x] = 1 / microns
            from astropy.units import micron

            if self.method.startswith('WD01'):
                self._tab_extinction = self._dustext_instance(self.tab_x / micron)
            else:
                raise NotImplemented('issue with E(B-V)-based lookup tab not resolved.')
                self._load()
        return self._tab_extinction

    @property
    def tab_attenuation(self):
        """
        Lookup table of attenuation vs wavelength.
        """
        if not hasattr(self, '_tab_attenuation'):
            # Callable expects [x] = 1 / microns
            from astropy.units import micron

            if self.method.startswith('WD01'):
                self._tab_attenuation = self._dustatt_instance(self.tab_x / micron)
            else:
                raise NotImplemented('issue with E(B-V)-based lookup tab not resolved.')
                self._load()
        return self._tab_attenuation

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

    def get_transmission(self, wave, Av=None, Sd=None, MUV=None, z=None):
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
            return 10**(-self.get_attenuation(wave, Av=Av, z=z) / 2.5)
        elif self.is_parameterized:
            tau = self.get_opacity(wave, Av=Av, Sd=Sd, z=z)
            return np.exp(-tau)
        elif self.is_irxb:
            # This case is handled separately by the `get_lf` method in
            # ares.populations objects.
            return 1.0
        else:
            raise NotImplemented('help')

    @property
    def tab_Av(self):
        if not hasattr(self, '_tab_Av'):
            self._tab_Av = np.arange(0, 10.1, 0.1)
        return self._tab_Av

    @property
    def _tab_C00(self):
        if not hasattr(self, '_tab_C00_'):
            if self.pf['pop_dust_cache'] is not None:
                self._tab_C00_ = self.pf['pop_dust_cache']
            else:
                self._tab_C00_ = np.zeros((self.tab_waves_c.size, self.tab_Av.size))
                for i, Av in enumerate(self.tab_Av):
                    C00 = self._dustatt_instance(Av=Av)
                    self._tab_C00_[:,i] = C00(self.tab_waves_c * 1e-4)
        return self._tab_C00_

    def get_curve(self, wave, z=None):
        """
        Get extinction (or attenuation) curve from lookup table.

        .. note :: This is what is contained in attenuation curves natively.

        Parameters
        ----------
        wave : int, float, np.ndarray
            Wavelength [Angstroms]

        """

        if self.is_template and self.method.startswith('C00'):
            # In this case, construct lookup table in Av, self.tab_waves_c
            iw = np.argmin(np.abs(wave - self.tab_waves_c))

            # Pretty crude for now
            A = self._tab_C00[iw,:]

        elif self.is_template:
            A = np.interp(wave, self.tab_waves_c, self.tab_extinction)
        else:
            raise NotImplemented('help')

        if self.pf['pop_dust_template_extension'] is not None:

            if not hasattr(self, '_get_temp_ext_'):
                self._get_temp_ext_ = get_function_from_par('pop_dust_template_extension',
                    self.pf)

            ext = self._get_temp_ext_(wave=wave, z=z)
            return A * ext
        else:
            return A

    def get_beta_from_MUV(self, MUV, z=None):
        assert self.is_irxb

        if type(self.pf['pop_muvbeta']) == str:
            assert self.pf['pop_muvbeta'] == 'bouwens2014', \
                "Only know Bouwens+ 2014 MUV-Beta right now!"

            zint = round(z, 0)

            if zint < 4:
                zint = 'lowz'
            elif zint >= 9:
                zint = 'highz'

            a, b = _coeff_b14[zint]

            return a + b * (MUV + 19.5)
        elif type(self.pf['pop_muvbeta']) == FunctionType:
            raise NotImplemented('help')
        elif type(self.pf['pop_muvbeta']) in numeric_types:
            beta = self.pf['pop_muvbeta']
        else:
            raise NotImplemented('help')

        return beta

    def get_AUV_from_irxb(self, MUV=None, beta=None, z=None):
        assert self.is_irxb

        if MUV is not None:
            beta = self.get_beta_from_MUV(MUV, z=z)

        # Currently only allow named IRX-Beta relations
        if type(self.pf['pop_irxbeta']) == str:
            assert self.pf['pop_irxbeta'] in _coeff_irxb.keys(), \
                f"Don't know {self.pf['pop_irxbeta']} IRX-Beta relation!"

            b, m = _coeff_irxb[self.pf['pop_irxbeta']]

            return np.maximum(b + m * beta, 0)
        elif type(self.pf['pop_irxbeta']) in [list, tuple, np.ndarray]:
            assert len(self.pf['pop_irxbeta']) == 2
            b, m = self.pf['pop_irxbeta']

            return np.maximum(b + m * beta, 0)
        else:
            raise NotImplemented('help')

    def get_attenuation(self, wave, Av=None, Sd=None, MUV=None, beta=None,
        z=None):
        if type(wave) in numeric_types:
            wave = np.array([wave])

        if self.is_template and self.method.startswith('C00'):
            if type(Av) in numeric_types:
                Av = np.array([Av])

            # This is a lookup table.
            if type(wave) in numeric_types:
                tab_A = self.get_curve(wave, z=z)
                A = np.interp(Av, self.tab_Av, tab_A, left=0)
            else:
                A = np.zeros((len(Av), wave.size))
                for i, _wave_ in enumerate(wave):
                    tab_A = self.get_curve(_wave_, z=z)
                    A[:,i] = np.interp(Av, self.tab_Av, tab_A, left=0)

        elif self.is_template:
            if type(Av) in numeric_types:
                Av = np.array([Av])
            A = self.get_curve(wave, z=z)[None,:] * Av[:,None]
        elif self.is_parameterized:
            if type(Sd) in numeric_types:
                Sd = np.array([Sd])
            tau = self.get_opacity(wave, Av=Av, Sd=Sd, z=z)
            A = 2.5 * tau / np.log(10.)
        elif self.is_irxb:
            assert 1000 <= wave <= 2000, \
                "Should only use this method for UV attenuation!"
            assert (MUV is not None) or (beta is not None), \
                "Must provide `MUV` or `beta`!"
            A = self.get_AUV_from_irxb(MUV=MUV, beta=beta, z=z)
        else:
            raise NotImplemented('help')

        if type(wave) in numeric_types:
            return A[:,0]
        else:
            return A

    def get_opacity(self, wave, Av=None, Sd=None, z=None):
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
            T = self.get_transmission(wave, Av=Av, Sd=Sd, z=z)
            tau = -np.log(T)
        elif self.is_parameterized:
            if type(Sd) in numeric_types:
                Sd = np.array([Sd])
            kappa = self.get_absorption_coeff(wave=wave, z=z)

            tau = kappa[None,:] * Sd[:,None]

            # Note that inf * 0 = NaN, which is a problem. This happens
            # sometimes, e.g., we set tau=inf in the Lyman continuum and then
            # we turn off dust by hand so Sd = 0, and we get nonsense answers.
            # Just check for that here.
            if np.any(np.isnan(tau)):
                ok_bad = np.logical_and(np.isinf(kappa[None,:]),
                                        Sd[:,None] == 0)
                tau[ok_bad==1] = 0
        else:
            raise NotImplemented('help')

        if wave.size == 1:
            return tau[:,0]
        else:
            return tau

    def get_absorption_coeff(self, wave, z=None):
        """
        Get dust absorption coefficient [cm^2 / g].
        """

        assert self.is_parameterized

        if not hasattr(self, '_get_kappa_'):
            self._get_kappa_ = get_function_from_par('pop_dust_absorption_coeff',
                self.pf)
        return self._get_kappa_(wave=wave)
