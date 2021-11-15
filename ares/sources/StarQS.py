"""

SingleStarModel.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Thu Jun  1 14:52:18 PDT 2017

Description: These are spectra from, e.g., Schaerer (2002), where we get a
series of Q values in different energy bins. Need to construct a continuous
spectrum to be used in RT calculations.

"""

import numpy as np
from .Star import Star
from .Source import Source
from ..physics import Cosmology
from scipy.integrate import quad
from ..util.ReadData import read_lit
from ..util.ParameterFile import ParameterFile
from ..physics.Constants import erg_per_ev, ev_per_hz, s_per_yr, g_per_msun, \
    c, h_p

class StarQS(Source):
    #def __init__(self, **kwargs):
    #    self.pf = ParameterFile(**kwargs)

    #@property
    #def cosm(self):
    #    if not hasattr(self, '_cosm'):
    #        self._cosm = Cosmology(**self.pf)
    #    return self._cosm

    @property
    def N(self):
        return self.PhotonsPerBaryon

    @property
    def Nion(self):
        if not hasattr(self, '_Nion'):
            self._Nion = self.N[1]
        return self._Nion

    @property
    def Nlw(self):
        if not hasattr(self, '_Nlw'):
            self._Nlw = self.N[0]
        return self._Nlw

    @property
    def PhotonsPerBaryon(self):
        return self.Q * self.lifetime * s_per_yr \
            / (self.pf['source_mass'] * g_per_msun * self.cosm.b_per_g)

    @property
    def ideal(self):
        if not hasattr(self, '_ideal'):
            self._ideal = Star(source_temperature=self.Teff,
                source_Emin=1, source_Emax=2e2,
                source_EminNorm=1, source_EmaxNorm=2e2)
        return self._ideal

    @property
    def name(self):
        return self.pf['source_sed']

    @property
    def litinst(self):
        if not hasattr(self, '_litinst'):
            self._litinst = read_lit(self.name)

        return self._litinst

    @property
    def bands(self):
        if not hasattr(self, '_bands'):
            if self.name == 'schaerer2002':
                self._bands = [(11.2, 13.6), (13.6, 24.6), (24.6, 54.4),\
                    (54.4, 2e2)]
            else:
                self._bands = None

        return self._bands

    @property
    def bandwidths(self):
        if not hasattr(self, '_bandwidths'):
            self._bandwidths = np.diff(self.bands, axis=1).squeeze()
        return self._bandwidths

    @property
    def Elo(self):
        if not hasattr(self, '_Elo'):
            _Elo = []
            for i, pt in enumerate(self.bands):
                _Elo.append(pt[0])
            self._Elo = np.array(_Elo)
        return self._Elo

    @property
    def Eavg(self):
        if not hasattr(self, '_Eavg'):
            self._Eavg = []
            for i, pt in enumerate(self.bands):
                self._Eavg.append(self.ideal.AveragePhotonEnergy(*pt))
            self._Eavg = np.array(self._Eavg)

        return self._Eavg

    @property
    def Q(self):
        if not hasattr(self, '_Q'):
            self._load()
        return self._Q

    @Q.setter
    def Q(self, value):
        self._Q = value

    @property
    def I(self):
        return self.Q * self.Eavg * erg_per_ev

    @property
    def Teff(self):
        if not hasattr(self, '_Teff'):
            self._load()

        return self._Teff

    @Teff.setter
    def Teff(self, value):
        self._Teff = value

    @property
    def lifetime(self):
        if not hasattr(self, '_lifetime'):
            self._load()
        return self._lifetime

    @lifetime.setter
    def lifetime(self, value):
        self._lifetime = value

    def _load(self):
        self._Q, self._Teff, self._lifetime = self.litinst._load(**self.pf)

    @property
    def norm_(self):
        if not hasattr(self, '_norm'):

            if self.pf['source_piecewise']:
                self._norm = []
                for i, band in enumerate(self.bands):
                    F = quad(lambda E: self.ideal.Spectrum(E), *band)[0]
                    self._norm.append(self.I[i] / F)
                self._norm = np.array(self._norm)
            else:
                band = (13.6, self.pf['source_Emax'])
                F = quad(lambda E: self.ideal.Spectrum(E), *band)[0]
                self._norm = np.array([np.sum(self.I[1:]) / F]*4)

        return self._norm

    @property
    def Lbol(self):
        if not hasattr(self, '_Lbol'):
            self._Lbol = np.sum(self.norm_)
        return self._Lbol

    def _SpectrumPiecewise(self, E):

        if E < self.bands[0][1]:
            return self.norm_[0] * self.ideal.Spectrum(E)
        elif self.bands[1][0] <= E < self.bands[1][1]:
            return self.norm_[1] * self.ideal.Spectrum(E)
        elif self.bands[2][0] <= E < self.bands[2][1]:
            return self.norm_[2] * self.ideal.Spectrum(E)
        else:
            return self.norm_[3] * self.ideal.Spectrum(E)

    @property
    def _spec_f(self):
        if not hasattr(self, '_spec_f_'):
            self._spec_f_ = np.vectorize(self._SpectrumPiecewise)
        return self._spec_f_

    def _Intensity(self, E, t=None):
        """
        Return quantity *proportional* to fraction of bolometric luminosity.

        Parameters
        ----------
        E : int, float
            Photon energy of interest [eV]

        """

        return self._spec_f(E)

    @property
    def norm(self):
        if not hasattr(self, '_norm_'):
            # Need
            integrand = lambda EE: self._Intensity(EE)
            self._norm_ = 1. / quad(integrand, *self.bands[0])[0]
        return self._norm_

    def Spectrum(self, E):
        return self.norm * self._Intensity(E)

    def rad_yield(self, Emin, Emax):
        """
        Must be in the internal units of erg / g.
        """

        return np.sum(self.I) * self.lifetime * s_per_yr \
            / (self.pf['source_mass'] * g_per_msun)

    @property
    def Emin(self):
        return self.Elo[0]

    @property
    def Emax(self):
        return self.bands[-1][1]

    def L_per_sfr(self, wave=1600., avg=1, Z=None, band=None, window=1,
            energy_units=True, raw=True, nebular_only=False):
        """
        Specific emissivity at provided wavelength.

        .. note :: This is analogous to SynthesisModel.L_per_sfr, but since
            in this class we deal with single stars (so ill-defined SFR),
            we're doing some gymnastics here to get things in the right units.
            All that really matters is that we have something, that when
            multiplied by an SFR, will yield a halo's luminosity in
            erg/s/Hz/(Msun/yr).

        Parameters
        ----------
        wave : int, float
            Wavelength at which to determine luminosity [Angstroms]
        avg : int
            Number of wavelength bins over which to average

        """

        sfr_eff = self.pf['source_mass'] / self.lifetime

        E = h_p * c / (wave * 1e-8) / erg_per_ev

        L = self.Lbol * self.Spectrum(E) * ev_per_hz / sfr_eff

        return L
