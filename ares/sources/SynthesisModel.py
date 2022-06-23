"""

SynthesisModel.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Mon Apr 11 11:27:45 PDT 2016

Description:

"""

import numpy as np
from ..data import ARES
from .Source import Source
from ..util.Math import interp1d
from ares.physics import Cosmology
from scipy.optimize import minimize
from scipy.integrate import cumtrapz
from ..util.ReadData import read_lit
from ..physics import NebularEmission
from ..util.ParameterFile import ParameterFile
from ares.physics.Constants import h_p, c, erg_per_ev, g_per_msun, s_per_yr, \
    s_per_myr, m_H, ev_per_hz, E_LL

class SynthesisModelBase(Source):
    @property
    def _nebula(self):
        if not hasattr(self, '_nebula_'):
            self._nebula_ = NebularEmission(cosm=self.cosm, **self.pf)
            self._nebula_.wavelengths = self.wavelengths
        return self._nebula_

    @property
    def _neb_cont(self):
        if not hasattr(self, '_neb_cont_'):
            self._neb_cont_ = np.zeros_like(self._data)
            if self.pf['source_nebular'] > 1 and \
                self.pf['source_nebular_continuum']:

                for i, t in enumerate(self.times):
                    if self.pf['source_tneb'] is not None:
                        j = np.argmin(np.abs(self.pf['source_tneb'] - self.times))
                    else:
                        j = i

                    spec = self._data_raw[:,j] * self.dwdn

                    # If is_ssp = False, should do cumulative integral
                    # over time here.

                    self._neb_cont_[:,i] = \
                        self._nebula.Continuum(spec) / self.dwdn

        return self._neb_cont_

    @property
    def _neb_line(self):
        if not hasattr(self, '_neb_line_'):
            self._neb_line_ = np.zeros_like(self._data)
            if self.pf['source_nebular'] > 1 and \
                self.pf['source_nebular_lines']:
                for i, t in enumerate(self.times):
                    if self.pf['source_tneb'] is not None:
                        j = np.argmin(np.abs(self.pf['source_tneb'] - self.times))
                    else:
                        j = i

                    spec = self._data_raw[:,j] * self.dwdn

                    self._neb_line_[:,i] = \
                        self._nebula.LineEmission(spec) / self.dwdn

        return self._neb_line_

    def _add_nebular_emission(self):
        # Keep raw spectrum
        self._data_raw = self._data.copy()

        # Add in nebular continuum (just once!)
        added_neb_cont = 0
        added_neb_line = 0
        null_ionizing_spec = 0
        if not hasattr(self, '_neb_cont_'):
            self._data += self._neb_cont
            added_neb_cont = 1

        # Same for nebular lines.
        if not hasattr(self, '_neb_line_'):
            self._data += self._neb_line
            added_neb_line = 1

        if added_neb_cont or added_neb_line:
            null_ionizing_spec = self.pf['source_nebular'] > 1

        if null_ionizing_spec:
            self._data[self.energies > E_LL] *= self.pf['source_fesc']

    def AveragePhotonEnergy(self, Emin, Emax):
        """
        Return average photon energy in supplied band.
        """

        j1 = np.argmin(np.abs(Emin - self.energies))
        j2 = np.argmin(np.abs(Emax - self.energies))

        E = self.energies[j2:j1][-1::-1]

        # Units: erg / s / Hz
        to_int = self.Spectrum(E)

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

    def Spectrum(self, E):
        """
        Return a normalized version of the spectrum at photon energy E / eV.
        """

        cached_result = self._cache_spec(E)
        if cached_result is not None:
            return cached_result

        # reverse energies so they are in ascending order
        nrg = self.energies[-1::-1]

        spec = np.interp(E, nrg, self.sed_at_tsf[-1::-1]) / self.norm

        if type(E) != np.ndarray:
            self._cache_spec_[E] = spec

        return spec

    def get_sed_at_t(self, t=None, i_tsf=None, raw=False, nebular_only=False):
        if i_tsf is None:
            i_tsf = np.argmin(np.abs(t - self.times))

        if raw and not (nebular_only or self.pf['source_nebular_only']):
            poke = self.sed_at_tsf
            data = self._data_raw
        else:
            data = self.data.copy()

            if nebular_only or self.pf['source_nebular_only']:
                poke = self.sed_at_tsf
                data -= self._data_raw

        # erg / s / Hz -> erg / s / eV
        if self.pf['source_rad_yield'] == 'from_sed':
            sed = data[:,i_tsf] * self.dwdn / ev_per_hz
        else:
            sed = data[:,i_tsf]

        return sed

    @property
    def sed_at_tsf(self):
        if not hasattr(self, '_sed_at_tsf'):
            self._sed_at_tsf = self.get_sed_at_t(i_tsf=self.i_tsf,
                raw=False)
        return self._sed_at_tsf

    @property
    def sed_at_tsf_raw(self):
        if not hasattr(self, '_sed_at_tsf_raw'):
            self._sed_at_tsf_raw = self.get_sed_at_t(i_tsf=self.i_tsf,
                raw=True)
        return self._sed_at_tsf_raw

    @property
    def dE(self):
        if not hasattr(self, '_dE'):
            tmp = np.abs(np.diff(self.energies))
            self._dE = np.concatenate((tmp, [tmp[-1]]))
        return self._dE

    @property
    def dndE(self):
        if not hasattr(self, '_dndE'):
            tmp = np.abs(np.diff(self.frequencies) / np.diff(self.energies))
            self._dndE = np.concatenate((tmp, [tmp[-1]]))
        return self._dndE

    @property
    def dwdn(self):
        if not hasattr(self, '_dwdn'):
            #if np.allclose(np.diff(np.diff(self.wavelengths)), 0):
            self._dwdn = self.wavelengths**2 / (c * 1e8)
            #else:
            #tmp = np.abs(np.diff(self.wavelengths) / np.diff(self.frequencies))
            #    self._dwdn = np.concatenate((tmp, [tmp[-1]]))
        return self._dwdn

    @property
    def norm(self):
        """
        Normalization constant that forces self.Spectrum to have unity
        integral in the (Emin, Emax) band.
        """
        if not hasattr(self, '_norm'):
            # Note that we're not using (EminNorm, EmaxNorm) band because
            # for SynthesisModels we don't specify luminosities by hand. By
            # using (EminNorm, EmaxNorm), we run the risk of specifying a
            # range not spanned by the model.
            j1 = np.argmin(np.abs(self.Emin - self.energies))
            j2 = np.argmin(np.abs(self.Emax - self.energies))

            # Remember: energy axis in descending order
            # Note use of sed_at_tsf_raw: need to be careful to normalize
            # to total power before application of fesc.
            self._norm = np.trapz(self.sed_at_tsf_raw[j2:j1][-1::-1],
                x=self.energies[j2:j1][-1::-1])

        return self._norm

    @property
    def i_tsf(self):
        if not hasattr(self, '_i_tsf'):
            self._i_tsf = np.argmin(np.abs(self.pf['source_tsf'] - self.times))
        return self._i_tsf

    @property
    def Nfreq(self):
        return len(self.energies)

    @property
    def E(self):
        if not hasattr(self, '_E'):
            self._E = np.sort(self.energies)
        return self._E

    @property
    def LE(self):
        """
        Should be dimensionless?
        """
        if not hasattr(self, '_LE'):
            if self.pf['source_ssp']:
                raise NotImplemented('No support for SSPs yet (due to t-dep)!')

            _LE = self.sed_at_tsf * self.dE / self.Lbol_at_tsf

            s = np.argsort(self.energies)
            self._LE = _LE[s]

        return self._LE

    @property
    def energies(self):
        if not hasattr(self, '_energies'):
            self._energies = h_p * c / (self.wavelengths / 1e8) / erg_per_ev
        return self._energies

    @property
    def Emin(self):
        return np.min(self.energies)

    @property
    def Emax(self):
        return np.max(self.energies)

    @property
    def frequencies(self):
        if not hasattr(self, '_frequencies'):
            self._frequencies = c / (self.wavelengths / 1e8)
        return self._frequencies

    @property
    def emissivity_per_sfr(self):
        """
        Photon emissivity.
        """
        if not hasattr(self, '_E_per_M'):
            self._E_per_M = np.zeros_like(self.data)
            for i in range(self.times.size):
                self._E_per_M[:,i] = self.data[:,i] \
                    / (self.energies * erg_per_ev)

        return self._E_per_M

    def get_beta(self, wave1=1600, wave2=2300, data=None):
        """
        Return UV slope in band between `wave1` and `wave2`.

        .. note :: This is just finite-differencing in log space. Should use
            routines in GalaxyEnsemble for more precision.
        """
        if data is None:
            data = self.data

        ok = np.logical_or(wave1 == self.wavelengths,
                           wave2 == self.wavelengths)

        arr = self.wavelengths[ok==1]

        Lh_l = np.array(data[ok==1,:])

        logw = np.log(arr)
        logL = np.log(Lh_l)

        return (logL[0,:] - logL[-1,:]) / (logw[0,None] - logw[-1,None])

    def LUV_of_t(self):
        return self.L_per_sfr_of_t()

    def _cache_L(self, wave, avg, Z, raw, nebular_only):
        if not hasattr(self, '_cache_L_'):
            self._cache_L_ = {}

        if (wave, avg, Z, raw, nebular_only) in self._cache_L_:
            return self._cache_L_[(wave, avg, Z, raw, nebular_only)]

        return None

    def L_per_sfr_of_t(self, wave=1600., avg=1, Z=None, units='Hz',
        raw=False, nebular_only=False):
        """
        UV luminosity per unit SFR.
        """

        cached_result = self._cache_L(wave, avg, Z, raw, nebular_only)

        if cached_result is not None:
            return cached_result

        if type(wave) in [list, tuple, np.ndarray]:

            E1 = h_p * c / (wave[0] * 1e-8) / erg_per_ev
            E2 = h_p * c / (wave[1] * 1e-8) / erg_per_ev

            yield_UV = self.IntegratedEmission(Emin=E2, Emax=E1,
                energy_units=True, raw=raw, nebular_only=nebular_only)

        else:
            j = np.argmin(np.abs(wave - self.wavelengths))

            if Z is not None:
                assert not raw, "Fix Z-dep option!"
                Zvals = np.sort(list(self.metallicities.values()))
                k = np.argmin(np.abs(Z - Zvals))
                raw = self.data # just to be sure it has been read in.
                data = self._data_all_Z[k,j]
            else:
                if raw and not (nebular_only or self.pf['source_nebular_only']):
                    poke = self.sed_at_tsf
                    data = self._data_raw[j,:]
                else:
                    data = self.data[j,:].copy()
                    if nebular_only or self.pf['source_nebular_only']:
                        poke = self.sed_at_tsf
                        data -= self._data_raw[j,:]

            if avg == 1:
                if units == 'Hz':
                    yield_UV = data * np.abs(self.dwdn[j])
                else:
                    yield_UV = data
            else:
                if Z is not None:
                    raise NotImplemented('hey!')
                assert avg % 2 != 0, "avg must be odd"
                avg = int(avg)
                s = (avg - 1) / 2

                j1 = np.argmin(np.abs(wave - s - self.wavelengths))
                j2 = np.argmin(np.abs(wave + s - self.wavelengths))

                if units == 'Hz':
                    yield_UV = np.mean(self.data[j1:j2+1,:] \
                        * np.abs(self.dwdn[j1:j2+1])[:,None], axis=0)
                else:
                    yield_UV = np.mean(self.data[j1:j2+1,:])

        # Current units:
        # if pop_ssp:
        #     erg / sec / Hz / Msun
        # else:
        #     erg / sec / Hz / (Msun / yr)

        self._cache_L_[(wave, avg, Z, units, raw, nebular_only)] = yield_UV

        return yield_UV

    def _cache_L_per_sfr(self, wave, avg, Z, raw, nebular_only):
        if not hasattr(self, '_cache_L_per_sfr_'):
            self._cache_L_per_sfr_ = {}

        if (wave, avg, Z, raw, nebular_only) in self._cache_L_per_sfr_:
            return self._cache_L_per_sfr_[(wave, avg, Z, raw, nebular_only)]

        return None

    def L_per_sfr(self, wave=1600., avg=1, Z=None, band=None, window=1,
            energy_units=True, raw=False, nebular_only=False):
        """
        Specific emissivity at provided wavelength at `source_tsf`.

        .. note :: This is just taking self.L_per_sfr_of_t and interpolating
            to some time, source_tsf. This is generally used when assuming
            constant star formation -- in the UV, L_per_sfr_of_t will
            asymptote to a ~constant value after ~100s of Myr.

        Parameters
        ----------
        wave : int, float
            Wavelength at which to determine emissivity.
        avg : int
            Number of wavelength bins over which to average

        Units are
            erg / s / Hz / (Msun / yr)
        or
            erg / s / Hz / Msun

        """

        cached = self._cache_L_per_sfr(wave, avg, Z, raw, nebular_only)

        if cached is not None:
            return cached

        yield_UV = self.L_per_sfr_of_t(wave, raw=raw, nebular_only=nebular_only)

        # Interpolate in time to obtain final LUV
        if self.pf['source_tsf'] in self.times:
            result = yield_UV[np.argmin(np.abs(self.times - self.pf['source_tsf']))]
        else:
            k = np.argmin(np.abs(self.pf['source_tsf'] - self.times))
            if self.times[k] > self.pf['source_tsf']:
                k -= 1

            func = interp1d(self.times, yield_UV, kind='linear')
            result = func(self.pf['source_tsf'])

        self._cache_L_per_sfr_[(wave, avg, Z, raw, nebular_only)] = result

        return result

    def integrated_emissivity(self, l0, l1, unit='A'):
        # Find band of interest -- should be more precise and interpolate

        if unit == 'A':
            x = self.wavelengths
            i0 = np.argmin(np.abs(x - l0))
            i1 = np.argmin(np.abs(x - l1))
        elif unit == 'Hz':
            x = self.frequencies
            i1 = np.argmin(np.abs(x - l0))
            i0 = np.argmin(np.abs(x - l1))

        # Current units: photons / sec / baryon / Angstrom

        # Count up the photons in each spectral bin for all times
        photons_per_b_t = np.zeros_like(self.times)
        for i in range(self.times.size):
            photons_per_b_t[i] = np.trapz(self.emissivity_per_sfr[i1:i0,i],
                x=x[i1:i0])

        t = self.times * s_per_myr

    def erg_per_phot(self, Emin, Emax):
        return self.eV_per_phot(Emin, Emax) * erg_per_ev

    def eV_per_phot(self, Emin, Emax):
        """
        Compute the average energy per photon (in eV) in some band.
        """

        i0 = np.argmin(np.abs(self.energies - Emin))
        i1 = np.argmin(np.abs(self.energies - Emax))

        # [self.data] = erg / s / A / [depends]

        # Must convert units
        E_tot = np.trapz(self.data[i1:i0,:].T * self.wavelengths[i1:i0],
            x=np.log(self.wavelengths[i1:i0]), axis=1)
        N_tot = np.trapz(self.data[i1:i0,:].T * self.wavelengths[i1:i0] \
            / self.energies[i1:i0] / erg_per_ev,
            x=np.log(self.wavelengths[i1:i0]), axis=1)

        if self.pf['source_ssp']:
            return E_tot / N_tot / erg_per_ev
        else:
            return E_tot[-1] / N_tot[-1] / erg_per_ev

    def rad_yield(self, Emin, Emax, raw=True):
        """
        Must be in the internal units of erg / g.
        """

        erg_per_variable = \
           self.IntegratedEmission(Emin, Emax, energy_units=True, raw=raw)

        if self.pf['source_ssp']:
            # erg / s / Msun -> erg / s / g
            return erg_per_variable / g_per_msun
        else:
            # erg / g
            return erg_per_variable[-1] * s_per_yr / g_per_msun

    @property
    def Lbol_at_tsf(self):
        if not hasattr(self, '_Lbol_at_tsf'):
            self._Lbol_at_tsf = self.Lbol(self.pf['source_tsf'])
        return self._Lbol_at_tsf

    def Lbol(self, t, raw=True):
        """
        Return bolometric luminosity at time `t`.

        Assume 1 Msun / yr SFR.
        """

        L = self.IntegratedEmission(energy_units=True, raw=raw)

        return np.interp(t, self.times, L)

    def IntegratedEmission(self, Emin=None, Emax=None, energy_units=False,
        raw=True, nebular_only=False):
        """
        Compute photons emitted integrated in some band for all times.

        Returns
        -------
        Integrated flux between (Emin, Emax) for all times in units of
        photons / sec / (Msun [/ yr]), unless energy_units=True, in which
        case its erg instead of photons.
        """

        # Find band of interest -- should be more precise and interpolate
        if Emin is None:
            Emin = np.min(self.energies)
        if Emax is None:
            Emax = np.max(self.energies)

        i0 = np.argmin(np.abs(self.energies - Emin))
        i1 = np.argmin(np.abs(self.energies - Emax))

        if i0 == i1:
            print("Emin={}, Emax={}".format(Emin, Emax))
            raise ValueError('Are EminNorm and EmaxNorm set properly?')

        if raw and not (nebular_only or self.pf['source_nebular_only']):
            poke = self.sed_at_tsf
            data = self._data_raw
        else:
            data = self.data.copy()

            if nebular_only or self.pf['source_nebular_only']:
                poke = self.sed_at_tsf
                data -= self._data_raw

        # Count up the photons in each spectral bin for all times
        flux = np.zeros_like(self.times)
        for i in range(self.times.size):
            if energy_units:
                integrand = data[i1:i0,i] * self.wavelengths[i1:i0]
            else:
                integrand = data[i1:i0,i] * self.wavelengths[i1:i0] \
                    / (self.energies[i1:i0] * erg_per_ev)

            flux[i] = np.trapz(integrand, x=np.log(self.wavelengths[i1:i0]))

        # Current units:
        # if pop_ssp: photons / sec / Msun
        # else: photons / sec / (Msun / yr)

        return flux

    @property
    def Nion(self):
        if not hasattr(self, '_Nion'):
            self._Nion = self.PhotonsPerBaryon(13.6, 24.6)
        return self._Nion

    @property
    def Nlw(self):
        if not hasattr(self, '_Nlw'):
            self._Nlw = self.PhotonsPerBaryon(10.2, 13.6)
        return self._Nlw

    def PhotonsPerBaryon(self, Emin, Emax, raw=True, return_all_t=False):
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
        An array with the same dimensions as ``self.times``, representing the
        cumulative number of photons emitted per stellar baryon of star formation
        as a function of time.

        """

        #assert self.pf['pop_ssp'], "Probably shouldn't do this for continuous SF."
        photons_per_s_per_msun = self.IntegratedEmission(Emin, Emax, raw=raw,
            energy_units=False)

        # Current units:
        # if pop_ssp:
        #     photons / sec / Msun
        # else:
        #     photons / sec / (Msun / yr)

        # Integrate (cumulatively) over time
        if self.pf['source_ssp']:
            photons_per_b_t = photons_per_s_per_msun / self.cosm.b_per_msun
            if return_all_t:
                return cumtrapz(photons_per_b_t, x=self.times*s_per_myr,
                    initial=0.0)
            else:
                return np.trapz(photons_per_b_t, x=self.times*s_per_myr)
        # Take steady-state result
        else:
            photons_per_b_t = photons_per_s_per_msun * s_per_yr \
                / self.cosm.b_per_msun

            # Return last element: steady state result
            return photons_per_b_t[-1]

class SynthesisModel(SynthesisModelBase):
    #def __init__(self, **kwargs):
    #    self.pf = ParameterFile(**kwargs)

    @property
    def _litinst(self):
        if not hasattr(self, '_litinst_'):
            self._litinst_ = read_lit(self.pf['source_sed'])

        return self._litinst_

    @property
    def wavelengths(self):
        if not hasattr(self, '_wavelengths'):
            if self.pf['source_sed_by_Z'] is not None:
                self._wavelengths, junk = self.pf['source_sed_by_Z']
            else:
                data = self.data

        return self._wavelengths

    @property
    def weights(self):
        return self._litinst.weights

    @property
    def times(self):
        if not hasattr(self, '_times'):
            self._times = self._litinst.times
        return self._times

    @property
    def metallicities(self):
        return self._litinst.metallicities

    @property
    def data(self):
        """
        Units = erg / s / A / [depends]

        Where, if instantaneous burst, [depends] = 1e6 Msun
        and if continuous SF, [depends] = Msun / yr

        In SSP case, remove factor of 1e6 here so it propagates everywhere
        else.

        """

        if not hasattr(self, '_data'):

            if self.pf['source_sps_data'] is not None:
                _Z, _ssp, _waves, _times, _data = self.pf['source_sps_data']
                assert _Z == self.pf['source_Z']
                assert _ssp == self.pf['source_ssp']
                self._data = _data
                self._times = _times
                self._wavelengths = _waves
                self._add_nebular_emission()
                return self._data

            Zall_l = list(self.metallicities.values())
            Zall = np.sort(Zall_l)

            # Check to see dimensions of tmp. Depending on if we're
            # interpolating in Z, it might be multiple arrays.
            if (self.pf['source_Z'] in Zall_l):
                if self.pf['source_sed_by_Z'] is not None:
                    _tmp = self.pf['source_sed_by_Z'][1]
                    self._data = _tmp[np.argmin(np.abs(Zall - self.pf['source_Z']))]
                else:
                    self._wavelengths, self._data, _fn = \
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
                    self._wavelengths, _tmp, _fn = \
                        self._litinst._load(**self.pf)

                    if self.pf['verbose']:
                        for _fn_ in _fn:
                            print("# Loaded {}".format(_fn_.replace(ARES, '$ARES')))

                # Shape is (Z, wavelength, time)?
                to_interp = np.array(_tmp)
                self._data_all_Z = to_interp

                # If outside table's metallicity range, just use endpoints
                if self.pf['source_Z'] > max(Zall):
                    _raw_data = np.log10(to_interp[-1])
                elif self.pf['source_Z'] < min(Zall):
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
                        _raw_data[:,i] = inter(np.log10(self.pf['source_Z']))

                self._data = _raw_data

                # By doing the interpolation in log-space we sometimes
                # get ourselves into trouble in bins with zero flux.
                # Gotta correct for that!
                #self._data[np.argwhere(np.isnan(self._data))] = 0.0

            # Normalize by SFR or cluster mass.
            if self.pf['source_ssp']:
                # The factor of a million is built-in to the lookup tables
                self._data *= self.pf['source_mass'] / 1e6
                if hasattr(self, '_data_all_Z'):
                    self._data_all_Z *= self.pf['source_mass'] / 1e6
            else:
                #raise NotImplemented('need to revisit this.')
                self._data *= self.pf['source_sfr']

            self._add_nebular_emission()

        return self._data
