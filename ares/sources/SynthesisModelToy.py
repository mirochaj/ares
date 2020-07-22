"""

SynthesisModelToy.py

Author: Jordan Mirocha
Affiliation: McGill
Created on: Mon 20 May 2019 15:08:49 EDT

Description:

"""

import numpy as np
from .Source import Source
from ..util.ParameterFile import ParameterFile
from ..physics.Constants import c, h_p, erg_per_ev, cm_per_ang

class SynthesisModelToy(Source):
    def __init__(self, **kwargs):
        Source.__init__(self, **kwargs)
        #self.pf = ParameterFile(**kwargs)

        #self.Emin = self.pf['source_Emin']
        #self.Emax = self.pf['source_Emax']

    @property
    def energies(self):
        if not hasattr(self, '_energies'):
            if (self.pf['source_wavelengths'] is not None) or \
               (self.pf['source_lmin'] is not None):
                self._energies = h_p * c / self.wavelengths / cm_per_ang
            else:
                dE = self.pf['source_dE']
                dl = self.pf['source_dlam']
                if dE is not None:
                    self._energies = np.arange(self.Emin, self.Emax+dE, dE)
                elif dl is not None:
                    self._energies = h_p * c / self.wavelengths / cm_per_ang
                else:
                    raise ValueError('help')

            assert (dE is not None) + (dl is not None) == 1

        return self._energies

    @property
    def wavelengths(self):
        if not hasattr(self, '_wavelengths'):
            if self.pf['source_wavelengths'] is not None:
                self._wavelengths = self.pf['source_wavelengths']
            elif self.pf['source_lmin'] is not None:
                self._wavelengths = np.arange(self.pf['source_lmin'],
                    self.pf['source_lmax']+self.pf['source_lres'],
                    self.pf['source_lres'])
            else:
                dE = self.pf['source_dE']
                dl = self.pf['source_dlam']
                if dE is not None:
                    self._wavelengths = h_p * c / self.energies / erg_per_ev \
                        / cm_per_ang
                else:
                    w1 = h_p * c / self.Emin / erg_per_ev / cm_per_ang
                    w2 = h_p * c / self.Emax / erg_per_ev / cm_per_ang
                    self._wavelengths = np.arange(w2, w1+dl, dl)

            if (self._wavelengths.max() < 2e3):
                raise ValueError('Wavelengths all FUV. This is generally not a good idea!')

        return self._wavelengths

    @property
    def frequencies(self):
        if not hasattr(self, '_frequencies'):
            self._frequencies = c / self.wavelengths / cm_per_ang
        return self._frequencies

    @property
    def times(self):
        if not hasattr(self, '_times'):
            if self.pf['source_times'] is not None:
                self._times = self.pf['source_times']
            else:
                # Standard SPS time gridding
                self._times = 10**np.arange(0, 4.1, 0.1)
        return self._times

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
            self._dwdn =  self.wavelengths**2 / (c * 1e8)
        return self._dwdn

    def _Spectrum(self, t, wave=1600.):


        beta = self.pf["source_toysps_beta"]
        _norm = self.pf["source_toysps_norm"]
        gamma = self.pf["source_toysps_gamma"]
        delta = self.pf["source_toysps_delta"]
        alpha = self.pf["source_toysps_alpha"]
        trise = self.pf['source_toysps_trise']
        _t0 = self.pf['source_toysps_t0']
        lmin = self.pf['source_toysps_lmin']

        ok = wave >= lmin

        # Normalization of each wavelength is set by UV slope
        norm = _norm * (wave / 1600.)**beta

        # Assume that all wavelengths initially decline as a power-law
        # with the same index
        _gamma_ = gamma * (wave / 1600.)**delta
        pl_decay = (t / 1.)**_gamma_

        # Assume an exponential decay at some critical (wavelength-dependent)
        # timescale.
        t0 = _t0 * (wave / 1600.)**alpha
        exp_decay = np.exp(-t / t0)

        # Put it all together.
        spec = norm * pl_decay * exp_decay
        spec[ok==0] = 0

        # Assume log-linear at t < trise
        if t < trise:
            spec *= (t / trise)**1.5

        return spec


    @property
    def data(self):
        """
        Units of erg / s / A / Msun
        """
        if not hasattr(self, '_data'):
            self._data = np.zeros((self.wavelengths.size, self.times.size))
            for i, t in enumerate(self.times):
                self._data[:,i] = self._Spectrum(t, wave=self.wavelengths)

        return self._data


    def _cache_L(self, wave, avg, Z):
        if not hasattr(self, '_cache_L_'):
            self._cache_L_ = {}

        if (wave, avg, Z) in self._cache_L_:
            return self._cache_L_[(wave, avg, Z)]

        return None

    def L_per_sfr(self):
        raise NotImplemented("Shouldn't be necessary?")

    def L_per_sfr_of_t(self, wave=1600., avg=1, Z=None):
        """
        UV luminosity per unit SFR.
        """

        cached_result = self._cache_L(wave, avg, Z)

        if cached_result is not None:
            return cached_result

        j = np.argmin(np.abs(wave - self.wavelengths))

        if Z is not None:
            raise NotImplemented('no support for Z yet.')
            #Zvals = np.sort(self.metallicities.values())
            #k = np.argmin(np.abs(Z - Zvals))
            #raw = self.data # just to be sure it has been read in.
            #data = self._data_all_Z[k,j]
        else:
            data = self.data[j,:]

        if avg == 1:
            yield_UV = data * np.abs(self.dwdn[j])
        else:
            if Z is not None:
                raise NotImplemented('hey!')
            assert avg % 2 != 0, "avg must be odd"
            s = (avg - 1) / 2
            yield_UV = np.mean(self.data[j-s:j+s,:] * np.abs(self.dwdn[j-s:j+s]))

        # Current units:
        # if pop_ssp:
        #     erg / sec / Hz / Msun
        # else:
        #     erg / sec / Hz / (Msun / yr)

        # to erg / s / A / Msun
        #if self.pf['source_ssp']:
        #    yield_UV /= 1e6
        ## or erg / s / A / (Msun / yr)
        #else:
        #    pass

        self._cache_L_[(wave, avg, Z)] = yield_UV

        return yield_UV
