"""

Galaxy.py

Author: Jordan Mirocha
Affiliation: JPL / Caltech
Created on: Fri Jun 30 16:08:51 CEST 2023

Description:

"""

import numpy as np
from scipy.optimize import fmin
from ..core import SpectralSynthesis
from ..physics.Constants import s_per_myr
from .SynthesisModel import SynthesisModel

class Galaxy(SynthesisModel):
    """
    Class to handle phenomenological SFH models, e.g., delayed tau, exponential.
    """

    @property
    def synth(self):
        if not hasattr(self, '_synth'):
            self._synth = SpectralSynthesis(**self.pf)
            self._synth.src = self
            #self._synth._src_csfr = self._src_csfr
            self._synth.oversampling_enabled = self.pf['pop_ssp_oversample']
            self._synth.oversampling_below = self.pf['pop_ssp_oversample_age']
            self._synth.careful_cache = self.pf['pop_synth_cache_level']

        return self._synth

    def get_sfr(self, t, norm=1, tau=1e3):
        """
        Return the star formation rate at time (since Big Bang) `t` [in Myr].

        Parameters
        ----------
        t : int, float
        norm :
        """

        if self.pf['source_sfh'] == 'exp_decl':
            return norm * np.exp(-t / tau)
        else:
            raise NotImplemented('help')

    def get_kwargs(self, t, mass, sfr):
        """
        Determine the free parameters of a model needed to produce stellar mass
        `mass` and star formation rate `sfr` at `t` [since Big Bang / Myr].
        """

        if self.pf['source_sfh'] == 'exp_decl':
            # Almost analytic here
            sSFR = lambda logtau: 1. / (10**logtau * (np.exp(t / 10**logtau) - 1.))
            func = lambda tau: np.abs(np.log10(sSFR(tau) / (sfr / mass)))
            tau = 10**fmin(func, 3., disp=False)[0]
            norm = mass / tau / (1 - np.exp(-t / tau))
            kw = {'norm': norm, 'tau': tau}
        else:
            raise NotImplemented('help')

        return kw

    def get_sfh(self, t, mass, sfr, **kwargs):
        """
        Return the SFR over all times, provided a boundary condition on the
        stellar mass and SFR at some time `t`.
        """

        if self.pf['source_sfh'] == 'exp_decl':
            kw = self.get_kwargs(t, mass, sfr)
            return self.get_sfr(self.tab_t_pop, **kw)
        else:
            raise NotImplemented('help')

    def get_mass(self, t, **kwargs):
        """
        Return stellar mass for a given SFH model, integrate analytically
        if possible.
        """

        if self.pf['source_sfh'] == 'exp_decl':
            tau = kwargs['tau']
            norm = kwargs['norm']

            # Factor of 1e6 is to convert Myr -> years
            return norm * 1e6 * tau * (1. - np.exp(-t / tau))
        else:
            raise NotImplemented('help')

    def get_spec(self, zobs, t, mass, sfr, waves=None):
        """
        Return the rest-frame spectrum of a galaxy at given t, that has
        stellar mass `mass` and SFR `sfr`.

        Parameters
        ----------
        t : np.ndarray
            Array of times in Myr since Big Bang.
        """

        if waves is None:
            waves = self.tab_waves_c


        sfh = self.get_sfh(t, mass, sfr)

        tasc = self.tab_t_pop[-1::-1]
        sfh_asc = sfh[-1::-1]

        spec = self.synth.get_spec_rest(sfh=sfh_asc, tarr=tasc,
            waves=waves, zobs=zobs, load=False)
                #hist={'SFR': sfh, 't': t})

        return spec

    def get_spec_obs(self):
        pass

    def get_mags(self):
        pass

    def get_lum_per_sfr(self):
        pass

    def generate_sed_tables(self):
        pass
