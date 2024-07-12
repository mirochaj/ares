"""

SynthesisModel.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Mon Apr 11 11:27:45 PDT 2016

Description:

"""
from functools import cached_property

import numbers
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import cumtrapz

from ..core import SpectralSynthesis
from ..data import ARES
from .Source import Source
from ..util.Stats import bin_c2e
from ..util.Math import interp1d
from ..util.Misc import numeric_types
from ..physics import Cosmology
from ares.data import read as read_lit
from ..physics import NebularEmission
from ..util.ParameterFile import ParameterFile
from ..physics.Constants import (
    h_p,
    c,
    erg_per_ev,
    g_per_msun,
    s_per_yr,
    s_per_myr,
    m_H,
    ev_per_hz,
    E_LL,
)

class SynthesisModelBase(Source):
    @property
    def _nebula(self):
        if not hasattr(self, '_nebula_'):
            self._nebula_ = NebularEmission(cosm=self.cosm, **self.pf)
            self._nebula_.tab_waves_c = self._tab_waves_c
            self._nebula_.tab_waves_e = self.tab_waves_e
        return self._nebula_

    def _get_continuum_emission(self, data):
        #if not hasattr(self, '_neb_cont_'):
        self._neb_cont = np.zeros_like(data)
        if self.pf['source_nebular'] > 1 and \
            self.pf['source_nebular_continuum']:

            for i, t in enumerate(self.tab_t):
                if self.pf['source_tneb'] is not None:
                    j = np.argmin(np.abs(self.pf['source_tneb'] - self.tab_t))
                else:
                    j = i

                spec = data[:,j] * self.tab_dwdn

                # If is_ssp = False, should do cumulative integral
                # over time here.

                self._neb_cont[:,i] = \
                    self._nebula.Continuum(spec) / self.tab_dwdn

        return self._neb_cont

    def _get_line_emission(self, data):
        #if not hasattr(self, '_neb_line_'):
        self._neb_line = np.zeros_like(data)
        if self.pf['source_nebular'] > 1 and \
            self.pf['source_nebular_lines']:
            for i, t in enumerate(self.tab_t):
                if self.pf['source_tneb'] is not None:
                    j = np.argmin(np.abs(self.pf['source_tneb'] - self.tab_t))
                else:
                    j = i

                spec = data[:,j] * self.tab_dwdn

                self._neb_line[:,i] = \
                    self._nebula.get_line_emission(spec) / self.tab_dwdn

        return self._neb_line

    def _add_nebular_emission(self, data):

        if self._added_nebular_emission:
            raise AttributeError('Already added nebular emission!')

        # Keep raw spectrum
        self._data_raw = data.copy()

        # Add in nebular continuum (just once!)
        added_neb_cont = 0
        added_neb_line = 0
        null_ionizing_spec = 0
        #if not hasattr(self, '_neb_cont_'):
        if (self.pf['source_nebular'] > 1) and self.pf['source_nebular_continuum']:
            data += self._get_continuum_emission(data)
            added_neb_cont = 1

        # Same for nebular lines.
        #if not hasattr(self, '_neb_line_'):
        if self.pf['source_nebular'] > 1 and self.pf['source_nebular_lines']:
            data += self._get_line_emission(data)
            added_neb_line = 1

        if added_neb_cont or added_neb_line:
            null_ionizing_spec = self.pf['source_nebular'] > 1

        if null_ionizing_spec:
            data[self.tab_energies_c > E_LL] *= self.pf['source_fesc']

        self._added_nebular_emission = True

    def get_avg_photon_energy(self, band, band_units='eV'):
        """
        Return average photon energy in supplied band at `source_age`.

        Parameters
        ----------
        band : 2-element tuple, list, np.ndarray
            Wavelengths or photon energies that define the boundaries of the
            interval we care about.
        band_units : str
            Determines whether user-supplied `band` values are in 'eV' or
            'Angstroms' (only two options for now).

        Returns
        -------
        Average energy of photons in supplied band in eV.

        """

        if band_units.lower() == 'ev':
            Emin, Emax = band
        elif band_units.lower().startswith('ang'):
            Emin = h_p * c / (band[1] * 1e-8) / erg_per_ev
            Emax = h_p * c / (band[0] * 1e-8) / erg_per_ev
        else:
            raise NotImplementedError('Unrecognized `band_units` option.')

        j1 = np.argmin(np.abs(Emin - self.tab_energies_c))
        j2 = np.argmin(np.abs(Emax - self.tab_energies_c))

        E = self.tab_energies_c[j2:j1][-1::-1]

        # Units: erg / s / Hz
        to_int = self.get_spectrum(E)

        # Units: erg / s
        return np.trapz(to_int * E, x=E) / np.trapz(to_int, x=E)

    def _cache_spec(self, E):
        if not hasattr(self, '_cache_spec_'):
            self._cache_spec_ = {}

        if type(E) == np.ndarray:
            pass
        else:
            if E in self._cache_spec_:
                return self._cache_spec_[E]

        return None

    def get_spectrum(self, x, t=None, units='eV'):
        """
        Return a normalized version of the spectrum at photon energy E / eV.
        """

        E = self.get_ev_from_x(x, units=units)

        cached_result = self._cache_spec(E)
        if cached_result is not None:
            return cached_result

        # reverse energies so they are in ascending order
        nrg = self.tab_energies_c[-1::-1]

        spec = np.interp(E, nrg, self.tab_sed_at_age[-1::-1]) / self._norm

        if type(E) != np.ndarray:
            self._cache_spec_[E] = spec

        return spec

    def get_sed_at_t(self, t=None, i_tsf=None, raw=False, nebular_only=False):
        """
        Retrieve the SED at a user specified time [Myr].

        .. note :: This is essentially relieving us of having to find the
            appropriate index in the 2-D SED table, tab_sed, and/or
            remembering the order of its dimensions.

        """

        # Find the index closest to the time requested by the user.
        if i_tsf is None:
            i_tsf = np.argmin(np.abs(t - self.tab_t))

        # Construct a modified version of the SED [optional]
        if raw and not (nebular_only or self.pf['source_nebular_only']):
            # Just need to make sure the _data_raw attribute exists
            poke = self.tab_sed
            data = self._data_raw
        else:
            data = self.tab_sed.copy()

            if nebular_only or self.pf['source_nebular_only']:
                # Just need to make sure the _data_raw attribute exists
                poke = self.tab_sed
                data -= self._data_raw

        # erg / s / Hz -> erg / s / eV
        if self.pf['source_rad_yield'] == 'from_sed':
            sed = data[:,i_tsf] * self.tab_dwdn / ev_per_hz
        else:
            sed = data[:,i_tsf]

        return sed

    @cached_property
    def tab_sed_at_age(self):
        self._sed_at_tsf = self.get_sed_at_t(i_tsf=self.i_tsf, raw=False)
        return self._sed_at_tsf

    @cached_property
    def tab_sed_at_age_raw(self):
        self._sed_at_tsf_raw = self.get_sed_at_t(i_tsf=self.i_tsf, raw=True)
        return self._sed_at_tsf_raw

    #@cached_property
    #def tab_dE(self):
    #    self._dE = np.diff(self.tab_energies_e)
    #    return self._dE

    #@cached_property
    #def tab_dndE(self):
    #    self._dndE = np.abs(np.diff(self.tab_freq_e) / np.diff(self.tab_energies_e))
    #    return self._dndE

    @cached_property
    def tab_dwdn(self):
        self._dwdn = self._tab_waves_c**2 / (c * 1e8)
        return self._dwdn

    @property
    def _norm(self):
        """
        Normalization constant that forces self.get_spectrum to have unity
        integral in the (Emin, Emax) band.
        """
        if not hasattr(self, '_norm_'):
            # Note that we're not using (EminNorm, EmaxNorm) band because
            # for SynthesisModels we don't specify luminosities by hand. By
            # using (EminNorm, EmaxNorm), we run the risk of specifying a
            # range not spanned by the model.
            j1 = np.argmin(np.abs(self.Emin - self.tab_energies_c))
            j2 = np.argmin(np.abs(self.Emax - self.tab_energies_c))

            # Remember: energy axis in descending order
            # Note use of sed_at_tsf_raw: need to be careful to normalize
            # to total power before application of fesc.
            self._norm_ = np.trapz(self.tab_sed_at_age_raw[j2:j1][-1::-1],
                x=self.tab_energies_c[j2:j1][-1::-1])

        return self._norm_

    @property
    def i_tsf(self):
        if not hasattr(self, '_i_tsf'):
            if type(self.pf['source_age']) == str:
                # Issues when source_age is, e.g., 'hubble'. In this case,
                # i_tsf will not actually be used?
                self._i_tsf = 0
            else:
                self._i_tsf = np.argmin(np.abs(self.pf['source_age'] - self.tab_t))

        return self._i_tsf

    #@property
    #def E(self):
    #    if not hasattr(self, '_E'):
    #        self._E = np.sort(self.tab_energies_c)
    #    return self._E

    @property
    def LE(self):
        """
        Should be dimensionless?
        """
        if not hasattr(self, '_LE'):
            if self.is_ssp:
                raise NotImplemented('No support for SSPs yet (due to t-dep)!')

            Lbol_at_tsf = self.get_lum_per_sfr(band=(None,None),
                band_units='eV')

            _LE = self.tab_sed_at_age * self.tab_dE / Lbol_at_tsf

            s = np.argsort(self.tab_energies_c)
            self._LE = _LE[s]

        return self._LE

    @cached_property
    def tab_energies_c(self):
        #if not hasattr(self, '_energies'):
        self._energies = h_p * c / (self.tab_waves_c / 1e8) / erg_per_ev
        return self._energies

    @cached_property
    def tab_energies_e(self):
        self._energies_e = h_p * c / (self.tab_waves_e / 1e8) / erg_per_ev
        return self._energies_e

    @property
    def Emin(self):
        return max(np.min(self.tab_energies_c), self.pf['source_Emin'])

    @property
    def Emax(self):
        return min(np.max(self.tab_energies_c), self.pf['source_Emax'])

    @cached_property
    def tab_freq_c(self):
        #if not hasattr(self, '_frequencies'):
        self._frequencies = c / (self.tab_waves_c / 1e8)
        return self._frequencies

    @cached_property
    def tab_freq_e(self):
        self._freq_e = c / (self.tab_waves_e / 1e8)
        return self._energies_e

    def get_beta(self, wave1=1600, wave2=2300, data=None):
        """
        Return UV slope in band between `wave1` and `wave2`.

        .. note :: This is just finite-differencing in log space. Should use
            routines in GalaxyEnsemble for more precision.
        """
        if data is None:
            data = self.tab_sed

        ok = np.logical_or(wave1 == self.tab_waves_c,
                           wave2 == self.tab_waves_c)

        arr = self.tab_waves_c[ok==1]

        Lh_l = np.array(data[ok==1,:])

        logw = np.log(arr)
        logL = np.log(Lh_l)

        return (logL[0,:] - logL[-1,:]) / (logw[0,None] - logw[-1,None])

    def _cache_L(self, kwds):
        if not hasattr(self, '_cache_L_'):
            self._cache_L_ = {}

        if kwds in self._cache_L_:
            return self._cache_L_[kwds]

        return None

    def get_lum_per_sfr_of_t(self, x=1600., window=1, band=None,
        units='Angstrom', units_out='erg/s/Hz', raw=False, nebular_only=False,
        Z=None):
        """
        Compute the luminosity per unit SFR (or mass, if source_ssp=True).

        Parameters
        ----------
        x : int, float
            Wavelength of interest [Angstroms].
        window : int
            Will compute luminosity averaged over this window, assumed to be
            a delta lambda width in Angstroms.
        band : tuple
            If provided, should be a range over which to integrate the spectrum,
            in units of Angstroms.

        Returns
        -------
        Units of output depend on input parameters:

        By default, luminosities are output in erg/s/Hz/SFR. If source_ssp=True,
        then it's erg/s/Hz/(stellar mass).

        If `band` is provided, luminosities are integrated, so we lose the Hz^-1
        units and just have luminosities in erg/s/SFR or erg/s/mass.

        If `units` is not 'Hz', then returned value will carry Angstrom^-1 units
        instead of Hz^-1 units.

        Finally, if `energy_units=False`, then we return the photon luminosity,
        i.e., the output is in photons/s/[Hz or Angstrom]/[SFR or stellar mass].
        """

        kwds = x, window, band, units, units_out, raw, nebular_only, Z

        cached_result = self._cache_L(kwds)
        if cached_result is not None:
            return cached_result

        if band is not None:

            E1, E2 = self.get_ev_from_x(band, units=units)
            if (E1 < np.min(self.tab_energies_c)) and (E2 < np.min(self.tab_energies_c)):
                return np.zeros_like(self.tab_t)

            i0 = np.argmin(np.abs(self.tab_energies_c - E1))
            i1 = np.argmin(np.abs(self.tab_energies_c - E2))

            if raw and not (nebular_only or self.pf['source_nebular_only']):
                poke = self.tab_sed_at_age
                data = self._data_raw
            else:
                data = self.tab_sed.copy()

                if nebular_only or self.pf['source_nebular_only']:
                    poke = self.tab_sed_at_age
                    data -= self._data_raw

            # Count up the photons in each spectral bin for all times
            yield_UV = np.zeros_like(self.tab_t)
            for i in range(self.tab_t.size):

                # Special treatment for narrow bands
                if (i0 == i1) or abs(i0 - i1) == 1:
                    l1, l2 = self.get_ang_from_x(band, units=units)
                    dlam = abs(l1 - l2)

                    if 'erg' in units_out.lower():
                        yield_UV[i] = data[i1,i] * dlam
                    else:
                        yield_UV[i] = data[i1,i] * dlam \
                            / (self.tab_energies_c[i1] * erg_per_ev)

                else:

                    if 'erg' in units_out.lower():
                        integrand = data[i1:i0,i] * self.tab_waves_c[i1:i0]
                    else:
                        integrand = data[i1:i0,i] * self.tab_waves_c[i1:i0] \
                            / (self.tab_energies_c[i1:i0] * erg_per_ev)

                    yield_UV[i] = np.trapz(integrand, x=np.log(self.tab_waves_c[i1:i0]))

        else:
            wave = self.get_ang_from_x(x, units=units)
            j = np.argmin(np.abs(wave - self.tab_waves_c))

            if Z is not None:
                assert not raw, "Fix Z-dep option!"
                Zvals = np.sort(list(self.tab_metallicities))
                k = np.argmin(np.abs(Z - Zvals))
                raw = self.tab_sed # just to be sure it has been read in.
                data = self._data_all_Z[k,j]
            else:
                if raw and not (nebular_only or self.pf['source_nebular_only']):
                    poke = self.tab_sed_at_age
                    data = self._data_raw[j,:]
                else:
                    data = self.tab_sed[j,:].copy()
                    if nebular_only or self.pf['source_nebular_only']:
                        poke = self.tab_sed_at_age
                        data -= self._data_raw[j,:]

            if window == 1:
                if 'hz' in units_out.lower():
                    yield_UV = data * np.abs(self.tab_dwdn[j])
                else:
                    yield_UV = data
            else:
                if Z is not None:
                    raise NotImplemented('hey!')
                assert window % 2 != 0, "window must be odd"
                window = int(window)
                s = (window - 1) / 2

                j1 = np.argmin(np.abs(wave - s - self.tab_waves_c))
                j2 = np.argmin(np.abs(wave + s - self.tab_waves_c))

                if 'Hz' in units_out:
                    yield_UV = np.mean(self.tab_sed[j1:j2+1,:] \
                        * np.abs(self.tab_dwdn[j1:j2+1])[:,None], axis=0)
                else:
                    yield_UV = np.mean(self.tab_sed[j1:j2+1,:], axis=0)

        # Current units:
        # if pop_ssp:
        #     erg / sec / Hz / Msun
        # else:
        #     erg / sec / Hz / (Msun / yr)

        self._cache_L_[kwds] = yield_UV

        return yield_UV

    def _cache_L_per_sfr(self, wave, window, band, Z, raw, nebular_only, age):
        if not hasattr(self, '_cache_L_per_sfr_'):
            self._cache_L_per_sfr_ = {}

        if (wave, window, band, Z, raw, nebular_only, age) in self._cache_L_per_sfr_:
            return self._cache_L_per_sfr_[(wave, window, band, Z, raw, nebular_only, age)]

        return None

    def get_lum_per_sfr(self, x=1600., window=1, Z=None, age=None, band=None,
        units='Angstroms', units_out='erg/s/Hz', raw=False, nebular_only=False):
        """
        Specific emissivity at provided wavelength at `source_age`.

        .. note :: This is just taking self.get_lum_per_sfr_of_t and interpolating
            to some time, `source_age`. This is generally used when assuming
            constant star formation -- in the UV, get_lum_per_sfr_of_t will
            asymptote to a ~constant value after ~100s of Myr.

        Parameters
        ----------
        wave : int, float
            Wavelength at which to determine emissivity.
        window : int
            Number of wavelength bins over which to average

        Units are
            erg / s / Hz / (Msun / yr)
        or
            erg / s / Hz / Msun

        """

        #cached = self._cache_L_per_sfr(x, window, band, units_out, Z, raw, nebular_only, age)

        #if cached is not None:
        #    return cached

        yield_UV = self.get_lum_per_sfr_of_t(x, band=band, units=units,
            raw=raw, nebular_only=nebular_only, units_out=units_out,
            window=window)

        if age is not None:
            t = age
        else:
            t = self.pf['source_age']

        # Interpolate in time to obtain final LUV
        if t in self.tab_t:
            result = yield_UV[np.argmin(np.abs(self.tab_t - t))]
        else:
            k = np.argmin(np.abs(t - self.tab_t))
            if self.tab_t[k] > t:
                k -= 1

            func = interp1d(self.tab_t, yield_UV, kind='linear',
                bounds_error=False, left=yield_UV[0], right=yield_UV[-1])
            result = func(t)

        #self._cache_L_per_sfr_[(x, window, band, Z, raw, nebular_only, age)] = result

        return result

    def erg_per_phot(self, Emin, Emax):
        return self.eV_per_phot(Emin, Emax) * erg_per_ev

    def eV_per_phot(self, Emin, Emax):
        """
        Compute the average energy per photon (in eV) in some band.
        """

        i0 = np.argmin(np.abs(self.tab_energies_c - Emin))
        i1 = np.argmin(np.abs(self.tab_energies_c - Emax))

        # [self.tab_sed] = erg / s / A / [depends]

        # Must convert units
        E_tot = np.trapz(self.tab_sed[i1:i0,:].T * self.tab_waves_c[i1:i0],
            x=np.log(self.tab_waves_c[i1:i0]), axis=1)
        N_tot = np.trapz(self.tab_sed[i1:i0,:].T * self.tab_waves_c[i1:i0] \
            / self.tab_energies_c[i1:i0] / erg_per_ev,
            x=np.log(self.tab_waves_c[i1:i0]), axis=1)

        if self.is_ssp:
            return E_tot / N_tot / erg_per_ev
        else:
            return E_tot[-1] / N_tot[-1] / erg_per_ev

    def get_rad_yield(self, band=None, units='eV', units_out='erg/s/Hz',
        raw=True):
        """
        The amount of radiative energy output in a given band [erg/s/[depends]].

        If a simple stellar population (source_ssp==True), [depends] = Msun,
        otherwise it's Msun/yr.
        """

        erg_per_variable = \
            self.get_lum_per_sfr_of_t(band=band, units=units,
            units_out=units_out, raw=raw)

        if self.is_ssp:
            # erg / s / Msun
            return np.interp(self.pf['source_age'], self.tab_t,
                erg_per_variable)
        else:
            # erg / Msun
            return erg_per_variable[-1]

    #@property
    #def Lbol_at_tsf(self):
    #    if not hasattr(self, '_Lbol_at_tsf'):
    #        self._Lbol_at_tsf = self.Lbol(self.pf['source_age'])
    #    return self._Lbol_at_tsf

    #def Lbol(self, t, raw=True):
    #    """
    #    Return bolometric luminosity at time `t`.

    #    Assume 1 Msun / yr SFR.
    #    """

    #    L = self.IntegratedEmission(energy_units=True, raw=raw)

    #    return np.interp(t, self.tab_t, L)

    #def IntegratedEmission(self, Emin=None, Emax=None, energy_units=False,
    #    raw=True, nebular_only=False):
    #    """
    #    Compute photons emitted integrated in some band for all times.

    #    Returns
    #    -------
    #    Integrated flux between (Emin, Emax) for all times in units of
    #    photons / sec / (Msun [/ yr]), unless energy_units=True, in which
    #    case its erg instead of photons.
    #    """

    #    # Find band of interest -- should be more precise and interpolate
    #    if Emin is None:
    #        Emin = np.min(self.tab_energies_c)
    #    if Emax is None:
    #        Emax = np.max(self.tab_energies_c)

    #    if (Emin < np.min(self.tab_energies_c)) and (Emax < np.min(self.tab_energies_c)):
    #        return np.zeros_like(self.tab_t)

    #    i0 = np.argmin(np.abs(self.tab_energies_c - Emin))
    #    i1 = np.argmin(np.abs(self.tab_energies_c - Emax))

    #    if i0 == i1:
    #        print("Emin={}, Emax={}".format(Emin, Emax))
    #        raise ValueError('Are EminNorm and EmaxNorm set properly?')

    #    if raw and not (nebular_only or self.pf['source_nebular_only']):
    #        poke = self.tab_sed_at_age
    #        data = self._data_raw
    #    else:
    #        data = self.tab_sed.copy()

    #        if nebular_only or self.pf['source_nebular_only']:
    #            poke = self.tab_sed_at_age
    #            data -= self._data_raw

    #    # Count up the photons in each spectral bin for all times
    #    flux = np.zeros_like(self.tab_t)
    #    for i in range(self.tab_t.size):
    #        if energy_units:
    #            integrand = data[i1:i0,i] * self.tab_waves_c[i1:i0]
    #        else:
    #            integrand = data[i1:i0,i] * self.tab_waves_c[i1:i0] \
    #                / (self.tab_energies_c[i1:i0] * erg_per_ev)

    #        flux[i] = np.trapz(integrand, x=np.log(self.tab_waves_c[i1:i0]))

    #    # Current units:
    #    # if pop_ssp: photons / sec / Msun
    #    # else: photons / sec / (Msun / yr)
    #    return flux

    @property
    def Nion(self):
        return self.get_Nion()

    @property
    def Nlw(self):
        return self.get_Nlw()

    def get_Nion(self):
        return self.get_nphot_per_baryon(band=(13.6, 24.6), units='eV')

    def get_Nlw(self):
        return self.get_nphot_per_baryon(band=(10.2, 13.6), units='eV')

    def get_nphot_per_baryon(self, band, units='eV', raw=True, return_all_t=False):
        """
        Compute the number of photons emitted per unit stellar baryon.

        ..note:: This integrand over the provided band, and cumulatively over time.

        Parameters
        ----------
        Emin : int, float
            Minimum rest-frame photon energy to consider [eV].
        Emax : int, float
            Maximum rest-frame photon energy to consider [eV].

        Returns
        -------
        An array with the same dimensions as ``self.tab_t``, representing the
        cumulative number of photons emitted per stellar baryon of star formation
        as a function of time.

        """

        if not hasattr(self, '_nphot_per_bar'):
            self._nphot_per_bar = {}

        if (band, raw, return_all_t) in self._nphot_per_bar:
            return self._nphot_per_bar[(band, raw, return_all_t)]

        #assert self.pf['pop_ssp'], "Probably shouldn't do this for continuous SF."
        photons_per_s_per_msun = self.get_lum_per_sfr_of_t(band=band,
            raw=raw, units_out='/s/Hz', units=units)

        # Current units:
        # if pop_ssp:
        #     photons / sec / Msun
        # else:
        #     photons / sec / (Msun / yr)

        # Integrate (cumulatively) over time
        if self.is_ssp:
            photons_per_b_t = photons_per_s_per_msun / self.cosm.b_per_msun
            if return_all_t:
                phot_per_b = cumtrapz(photons_per_b_t, x=self.tab_t*s_per_myr,
                    initial=0.0)
            else:
                phot_per_b = np.trapz(photons_per_b_t, x=self.tab_t*s_per_myr)
        # Take steady-state result
        else:
            photons_per_b_t = photons_per_s_per_msun * s_per_yr \
                / self.cosm.b_per_msun

            # Return last element: steady state result
            phot_per_b = photons_per_b_t[-1]

        self._nphot_per_bar[(band, raw, return_all_t)] = phot_per_b

        return phot_per_b

class SynthesisModel(SynthesisModelBase):
    #def __init__(self, **kwargs):
    #    self.pf = ParameterFile(**kwargs)

    @property
    def _litinst(self):
        if not hasattr(self, '_litinst_'):
            self._litinst_ = read_lit(self.pf['source_sed'])

        return self._litinst_

    @cached_property
    def tab_waves_c(self):
        #if not hasattr(self, '_tab_waves_cs'):
        if self.pf['source_sed_by_Z'] is not None:
            self._tab_waves_c, junk = self.pf['source_sed_by_Z']
        else:
            data = self.tab_sed_raw

        return self._tab_waves_c

    @cached_property
    def tab_waves_e(self):
        self._waves_e = bin_c2e(self.tab_waves_c)
        return self._waves_e

    @cached_property
    def tab_times(self):
        data = self.tab_sed_raw
        return self._tab_times

    @property
    def tab_t(self):
        return self.tab_times

    #@property
    #def weights(self):
    #    return self._litinst.weights

    #@cached_property
    #def tab_t(self):
    #    if not hasattr(self, '_times'):
    #        self._times = self._litinst.times
    #    return self._times

    @cached_property
    def tab_metallicities(self):
        return self._litinst.metallicities.values()

    @cached_property
    def tab_sed_raw(self):
        """
        Units = erg / s / A / [depends]

        Where, if instantaneous burst, [depends] = 1e6 Msun
        and if continuous SF, [depends] = Msun / yr

        In SSP case, remove factor of 1e6 here so it propagates everywhere
        else.

        """

        self._added_nebular_emission = False

        if self.pf['source_sps_data'] is not None:
            _Z, _ssp, _waves, _times, _data = self.pf['source_sps_data']
            if type(_Z) in numeric_types:
                assert _Z == self.pf['source_Z'], \
                    f"Cached Z={_Z}, from parameter file it's {self.pf['source_Z']}"
            if type(_ssp) in [int, bool]:
                assert _ssp == self.is_ssp, \
                    f"Cached ssp={_ssp}, from parameter file it's {self.is_ssp}"
            self._data = _data
            self._tab_times = _times
            self._tab_waves_c = _waves
            #print('# WARNING: If supplying source_sps_data, should include any nebular emission too!')
            #self._add_nebular_emission(self._data)
            return self._data

        Zall_l = list(self.tab_metallicities)
        Zall = np.sort(Zall_l)

        # Check to see dimensions of tmp. Depending on if we're
        # interpolating in Z, it might be multiple arrays.
        if (self.pf['source_Z'] in Zall_l):
            if self.pf['source_sed_by_Z'] is not None:
                _tmp = self.pf['source_sed_by_Z'][1]
                self._data = _tmp[np.argmin(np.abs(Zall - self.pf['source_Z']))]
            else:
                self._tab_waves_c, self._tab_times, self._data, _fn = \
                    self._litinst._load(**self.pf)

                if self.pf['verbose']:
                    print("# Loaded {}".format(_fn.replace(ARES,
                        '$ARES')))
        else:
            if self.pf['source_sed_by_Z'] is not None:
                _tmp = self.pf['source_sed_by_Z'][1]
                assert len(_tmp) == len(Zall)
            else:
                # Will load in all metallicities
                self._tab_waves_c, self._tab_times, _tmp, _fn = \
                    self._litinst._load(**self.pf)

                if self.pf['verbose']:
                    for _fn_ in _fn:
                        print("# Loaded {}".format(_fn_.replace(ARES, '$ARES')))

            # Shape is (Z, wavelength, time)?
            to_interp = np.array(_tmp)
            self._data_all_Z = to_interp

            is_num = isinstance(self.pf['source_Z'], numbers.Number)

            # If outside table's metallicity range, just use endpoints
            if is_num and self.pf['source_Z'] > max(Zall):
                _raw_data = np.log10(to_interp[-1])
            elif is_num and self.pf['source_Z'] < min(Zall):
                _raw_data = np.log10(to_interp[0])
            else:
                # At each time, interpolate between SEDs at two different
                # metallicities. Note: interpolating to log10(SED) caused
                # problems when nebular emission was on and when
                # starburst99 was being used (mysterious),
                # hence the log-linear approach here.
                _raw_data = np.zeros_like(to_interp[0])
                for i, t in enumerate(self._litinst.times):
                    inter = interp1d(np.log10(Zall),
                        to_interp[:,:,i], axis=0,
                        fill_value=0.0, kind=self.pf['interp_Z'])

                    # In the general case where metallicity varies, do we
                    # ever use this?
                    if is_num:
                        _raw_data[:,i] = inter(np.log10(self.pf['source_Z']))
                    else:
                        _raw_data[:,i] = inter(np.log10(Zall.min()))

            self._data = _raw_data

            # By doing the interpolation in log-space we sometimes
            # get ourselves into trouble in bins with zero flux.
            # Gotta correct for that!
            #self._data[np.argwhere(np.isnan(self._data))] = 0.0

        # Normalize by SFR or cluster mass.
        if self.is_ssp:
            # The factor of a million is built-in to the lookup tables
            self._data *= self.pf['source_mass'] / 1e6
            if hasattr(self, '_data_all_Z'):
                self._data_all_Z *= self.pf['source_mass'] / 1e6
        else:
            #raise NotImplemented('need to revisit this.')
            self._data *= self.pf['source_sfr']

        # Zero-out first and last entry in spectral dimension to avoid
        # erroneous 'extrapolation' outside the provided range.
        self._data[0,:] = 0
        self._data[-1,:] = 0

        return self._data

    @cached_property
    def tab_sed(self):
        # Add nebular emission (duh)
        self._tab_sed = self.tab_sed_raw.copy()
        self._add_nebular_emission(self._tab_sed)

        # Can reduce SED to *only* specific windows, e.g., to create a
        # "lines only" model (probably for intuition-building).
        if self.pf['source_sed_null_except'] is not None:
            iall = np.arange(0, self.tab_waves_c.size)
            tmp = self._tab_sed.copy()
            for chunk in self.pf['source_sed_null_except']:
                lo, hi, keep_cont = chunk

                # Need to be inclusive here. If the null_except window is
                # broader than the intrinsic resolution (controlled by
                # source_sed_degrade) then we could accidentally null out
                # the region of interest.
                ilo = np.argmin(np.abs(lo - self.tab_waves_c))
                if self.tab_waves_e[ilo] > lo:
                    ilo -= 1

                ihi = np.argmin(np.abs(hi - self.tab_waves_c))
                if self.tab_waves_e[ihi] < hi:
                    ihi += 1

                ok = np.logical_and(iall >= ilo, iall < ihi)
                #ok = np.logical_and(self.tab_waves_c >= lo,
                #                    self.tab_waves_c < hi)
                notok = np.logical_not(ok)
                self._tab_sed[notok==1,:] = 0

                if keep_cont:
                    pass
                else:
                    self._tab_sed[ok==1,:] = self._tab_sed[ok==1,:] \
                                        - self.tab_sed_raw[ok==1,:]

        return self._tab_sed
