"""

Source.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sun Jul 22 16:28:08 2012

Description: Initialize a radiation source.

"""

import os
import re
import numpy as np
from scipy.integrate import quad
from ..util import ParameterFile
from ..util.Misc import numeric_types
from functools import cached_property
from ..physics.Hydrogen import Hydrogen
from ..physics.Cosmology import Cosmology
from ..util.ParameterFile import ParameterFile
from ..core.IntegralTables import IntegralTable
from ..core.InterpolationTables import LookupTable
from ..util.SetDefaultParameterValues import SourceParameters, \
    CosmologyParameters
from ..physics.CrossSections import PhotoIonizationCrossSection as sigma_E
from ..physics.Constants import erg_per_ev, E_LL, s_per_myr, E_LyA, h_p, c

try:
    import h5py
except ImportError:
    pass

np.seterr(all='ignore')   # exp overflow occurs when integrating BB
                          # will return 0 as it should for x large

_sed_tabs = ['leitherer1999', 'eldridge2009', 'eldridge2017',
    'schaerer2002', 'hybrid',
    'bpass_v1', 'bpass_v2', 'starburst99', 'sps-toy']

class Source(object):
    def __init__(self, grid=None, cosm=None, logN=None, init_tabs=True,
        **kwargs):
        """
        Initialize a radiation source object.

        ..note:: This is inherited by all other ares.sources classes.

        Parameters
        ----------
        grid: rt1d.static.Grid.Grid instance
        logN: column densities over which to tabulate integral quantities

        """

        self.pf = ParameterFile(**kwargs)
        self._cosm_ = cosm

        # Create lookup tables for integral quantities
        if init_tabs and (grid is not None):
            self._create_integral_table(logN=logN)

    @cached_property
    def is_sed_tabular(self):
        return self.pf['source_sed'] in _sed_tabs

    @cached_property
    def is_ssp(self):
        return self.pf['source_ssp']

    @property
    def Emin(self):
        return self.pf['source_Emin']
    @property
    def Emax(self):
        return self.pf['source_Emax']

    @property
    def EminNorm(self):
        if not hasattr(self, '_EminNorm'):
            if self.pf['source_EminNorm'] == None:
                self._EminNorm = self.pf['source_Emin']
            else:
                self._EminNorm = self.pf['source_EminNorm']

        return self._EminNorm

    @property
    def EmaxNorm(self):
        if not hasattr(self, '_EmaxNorm'):
            if self.pf['source_EmaxNorm'] == None:
                self._EmaxNorm = self.pf['source_Emax']
            else:
                self._EmaxNorm = self.pf['source_EmaxNorm']

        return self._EmaxNorm

    @property
    def info(self):
        """
        Print info like Nlw etc in various units!
        """
        pass

    @property
    def is_delta(self):
        if not hasattr(self, '_is_delta'):
            self._is_delta = self.pf['source_sed'] == 'delta'
        return self._is_delta

    @property
    def has_nebular_lines(self):
        if not hasattr(self, '_has_nebular_lines'):
            self._has_nebular_lines = self.pf['source_nebular_lines'] > 0 \
                and self.pf['source_nebular'] > 0
        return self._has_nebular_lines

    def SourceOn(self, t):
        if t < self.tau:
            return True
        else:
            return False

    @property
    def tau(self):
        if not hasattr(self, '_tau'):
            self._tau = self.pf['source_lifetime'] * s_per_myr
        return self._tau

    @property
    def cosm(self):
        if not hasattr(self, '_cosm'):
            if self._cosm_ is not None:
                self._cosm = self._cosm_
            elif self.grid is not None:
                self._cosm = self.grid.cosm
            else:
                self._cosm = Cosmology(pf=self.pf, **self.pf)

        return self._cosm

    @property
    def multi_freq(self):
        if not hasattr(self, '_multi_freq'):
            self._multi_freq = self.discrete and not self.pf['source_multigroup']

        return self._multi_freq

    @property
    def multi_group(self):
        if not hasattr(self, '_multi_group'):
            self._multi_group = self.discrete and self.pf['source_multigroup']

        return self._multi_group

    @property
    def ionizing(self):
        # See if source emits ionizing photons
        # Should also be function of absorbers
        if not hasattr(self, '_ionizing'):
            self._ionizing = self.pf['source_Emax'] > E_LL

        return self._ionizing

    @property
    def grid(self):
        if not hasattr(self, '_grid'):
            self._grid = None

        return self._grid

    @grid.setter
    def grid(self, value):
        self._grid = value

    @property
    def discrete(self):
        if not hasattr(self, '_discrete'):
            self._discrete = (self.pf['source_E'] != None) or \
                (self.pf['source_sed'] in ['eldridge2009', 'eldridge2017',
                    'leitherer1999'])

        return self._discrete

    @property
    def continuous(self):
        if not hasattr(self, '_continuous'):
            self._continuous = not self.discrete

        return self._continuous

    @property
    def hydr(self):
        if not hasattr(self, '_hydr'):
            self._hydr = None

        return self._hydr

    @hydr.setter
    def hydr(self, value):
        self._hydr = value

    @property
    def frec(self):
        """
        Compute average recycling fraction (i.e., spectrum-weighted frec).
        """

        if self.hydr is None:
            return None

        n = np.arange(2, self.hydr.nmax)
        En = np.array(list(map(self.hydr.ELyn, n)))
        In = np.array(list(map(self.get_spectrum, En))) / En
        fr = np.array(list(map(self.hydr.frec, n)))

        return np.sum(fr * In) / np.sum(In)

    @property
    def intrinsic_hardening(self):
        if not hasattr(self, '_intrinsic_hardening'):
            if 'source_hardening' in self.pf:
                self._intrinsic_hardening = \
                    self.pf['source_hardening'] == 'intrinsic'
            else:
                self._intrinsic_hardening = False

        return self._intrinsic_hardening

    def _hardening_factor(self, E):
        return np.exp(-10.**self.logN \
            * (sigma_E(E, 0) + self.cosm.y * sigma_E(E, 1)))

    @property
    def logN(self):
        if not hasattr(self, '_logN'):
            if 'source_logN' in self.pf:
                self._logN = self.pf['source_logN']
            else:
                self._logN = -np.inf

        return self._logN

    @property
    def sharp_points(self):
        if not hasattr(self, '_sharp_points'):
            if self.pf['source_sed_sharp_at'] is not None:
                self._sharp_points = [self.pf['source_sed_sharp_at']]
            else:
                self._sharp_points = None

        return self._sharp_points

    @property
    def _normL(self):
        if not hasattr(self, '_normL_'):
            if self.is_delta:
                self._normL_ = 1. #/ self.pf['source_Emax']#/ self._Intensity(self.pf['source_Emax'])
            elif self.pf['source_Enorm'] is not None:
                En = self.pf['source_Enorm']

                if self.intrinsic_hardening:
                    self._normL_ = 1. / self._Intensity(En),
                else:
                    self._normL_ = 1. / (self._Intensity(En) / self._hardening_factor(En))
            else:
                if self.intrinsic_hardening:
                    self._normL_ = 1. / quad(self._Intensity,
                        self.pf['source_EminNorm'],
                        self.pf['source_EmaxNorm'], points=self.sharp_points)[0]
                else:
                    integrand = lambda EE: self._Intensity(EE) / self._hardening_factor(EE)
                    self._normL_ = 1. / quad(integrand,
                        self.pf['source_EminNorm'],
                        self.pf['source_EmaxNorm'], points=self.sharp_points)[0]

        return self._normL_

    #def _load_spectrum(self):
    #    """ Modify a few parameters if spectrum_file provided. """
    #
    #    fn = self.pf['spectrum_file']
    #
    #    if fn is None:
    #        return
    #
    #    # Read spectrum - expect hdf5 with (at least) E, LE, and t datasets.
    #    if re.search('.hdf5', fn):
    #        f = h5py.File(fn)
    #        try:
    #            self.pf['tables_times'] = f['t'].value
    #        except:
    #            self.pf['tables_times'] = None
    #            self.pf['spectrum_evolving'] = False
    #
    #        self.pf['spectrum_E'] = f['E'].value
    #        self.pf['spectrum_LE'] = f['LE'].value
    #        f.close()
    #
    #        if len(self.pf['spectrum_LE'].shape) > 1 \
    #            and not self.pf['spectrum_evolving']:
    #            self.pf['spectrum_LE'] = self.pf['spectrum_LE'][0]
    #    else:
    #        spec = readtab(fn)
    #        if len(spec) == 2:
    #            self.pf['spectrum_E'], self.pf['spectrum_LE'] = spec
    #        else:
    #            self.pf['spectrum_E'], self.pf['spectrum_LE'], \
    #                self.pf['spectrum_t'] = spec

    @property
    def tables(self):
        if not hasattr(self, '_tables'):
            self._create_integral_table()
        return self._tables

    @property
    def tab(self):
        if not hasattr(self, '_tab'):
            self._create_integral_table()
        return self._tab

    @property
    def tabs(self):
        if not hasattr(self, '_tabs'):
            self._create_integral_table()
        return self._tabs

    def _create_integral_table(self, logN=None):
        """
        Take tables and create interpolation functions.
        """

        if self.discrete:
            return

        if self._name == 'diffuse':
            return

        if self.pf['source_table'] is None:
            # Overide defaults if supplied - this is dangerous
            if logN is not None:
                self.pf.update({'tables_dlogN': [np.diff(tmp) for tmp in logN]})
                self.pf.update({'tables_logNmin': [np.min(tmp) for tmp in logN]})
                self.pf.update({'tables_logNmax': [np.max(tmp) for tmp in logN]})

            # Tabulate away!
            self._tab = IntegralTable(self.pf, self, self.grid, logN)
            self._tabs = self.tab.TabulateRateIntegrals()
        else:
            self._tab = IntegralTable(self.pf, self, self.grid, logN)
            self._tabs = self.tab.load(self.pf['source_table'])

        self._setup_interp()

    def _setup_interp(self):
        self._tables = {}
        for tab in self.tabs:
            self._tables[tab] = \
                LookupTable(self.pf, tab, self.tab.logN, self.tabs[tab],
                    self.tab.logx, self.tab.t)

    @property
    def sigma(self):
        """
        Compute bound-free absorption cross-section for all frequencies.
        """
        if not self.discrete:
            return None
        if not hasattr(self, '_sigma_all'):
            self._sigma_all = np.array(list(map(sigma_E, self.E)))

        return self._sigma_all

    def Qdot(self, t=None):
        """
        Returns number of photons emitted (s^-1) at all frequencies.
        """
        #if not hasattr(self, '_Qdot_all'):
        self._Qdot_all = self.Lbol(t) * self.LE / self.E / erg_per_ev

        return self._Qdot_all

    def hnu_bar(self, t=0):
        """
        Average ionizing (per absorber) photon energy in eV.
        """
        if not hasattr(self, '_hnu_bar_all'):
            self._hnu_bar_all = {}
        if not hasattr(self, '_qdot_bar_all'):
            self._qdot_bar_all = {}

        if t in self._hnu_bar_all:
            return self._hnu_bar_all[t]

        self._hnu_bar_all[t] = np.zeros_like(self.grid.zeros_absorbers)
        self._qdot_bar_all[t] = np.zeros_like(self.grid.zeros_absorbers)
        for i, absorber in enumerate(self.grid.absorbers):
            self._hnu_bar_all[t][i], self._qdot_bar_all[t][i] = \
                self._FrequencyAveragedBin(absorber=absorber, t=t)

        return self._hnu_bar_all

    def get_avg_photon_energy(self, Emin, Emax):
        """
        Return average photon energy in supplied band.
        """

        integrand = lambda EE: self.get_spectrum(EE) * EE
        norm = lambda EE: self.get_spectrum(EE)

        return quad(integrand, Emin, Emax, points=self.sharp_points)[0] \
             / quad(norm, Emin, Emax, points=self.sharp_points)[0]

    def get_ang_from_x(self, x, units='eV'):
        xout = self.get_ev_from_x(x, units=units)
        return h_p * c / erg_per_ev / xout / 1e-8

    def get_ev_from_x(self, x, units='eV'):
        """
        Convert input `x` from `units` to electron volts.
        """
        type_in = type(x)
        if type_in in [list, tuple]:
            x = np.array(x)
        elif type_in in numeric_types:
            x = np.array([x])

        if units.lower() == 'ev':
            xout = x
        elif units.lower().startswith('ang'):
            xout = h_p * c / erg_per_ev / x / 1e-8
        elif units.lower().startswith('mic'):
            xout = h_p * c / erg_per_ev / x / 1e-4
        elif units.lower().startswith('hz'):
            xout = h_p * x / erg_per_ev
        else:
            raise NotImplemented('help ')

        # Re-order if necessary
        if x.size > 1:
            if xout[0] > xout[1]:
                xout = np.flip(xout)

        if type_in == tuple:
            return tuple(xout)
        elif type_in == list:
            return list(xout)
        elif type_in in numeric_types:
            return float(xout)
        else:
            return xout

    def get_band_name(self, x=None, band=None, units='eV'):
        """
        Some bandpasses get special treatment, e.g., Lyman-Werner,
        Lyman-continuum, X-ray.


        Parameters
        ----------
        """

        monochromatic = False

        # Convert from monochromatic wavelength/energy/frequency to band
        if band is None:
            assert x is not None, "Must supply `x` or `band`!"
            monochromatic = True
            band = x, x

        # Unpack band and convert units to eV for band identification
        Emin, Emax = self.get_ev_from_x(band, units=units)

        # Easier case, check that energy/wave/freq lies in given interval.
        if monochromatic:
            if Emin >= self.pf['pop_Emin_xray']:
                name = 'Xray'
            elif Emax < E_LyA:
                name = 'OIR'
            elif (Emin > E_LyA) and (Emin < E_LL):
                name = 'LW'
            elif (Emin > E_LL):
                name = 'LyC'
            else:
                name = None

            return name

        # Slightly harder case, user provides bands. More annoying because
        # we're sloppy with sigfigs, check for variations of the same thing.
        if Emin >= self.pf['pop_Emin_xray']:
            name = 'Xray'
        elif Emax < E_LyA:
            name = 'OIR'
        elif (Emin, Emax) in [(E_LL, 24.6), (13.6, 24.6)]:
            name = 'LyC'
        elif (Emin > E_LL) and (Emax < self.pf['pop_Emin_xray']):
            name = 'LyC'
        elif (Emin, Emax) in [(10.2, 13.6), (11.2, 13.6),
            (10.2, E_LL), (11.2, E_LL), (E_LyA, 13.6), (E_LyA, E_LL)]:
            name = 'LW'
        elif (Emin > E_LyA) and (Emax < E_LL):
            name = 'LW'
        else:
            name = None

        return name

    @property
    def qdot_bar(self):
        """
        Average ionizing photon luminosity (per absorber) in s^-1.
        """
        if not hasattr(self, '_qdot_bar_all'):
            hnu_bar = self.hnu_bar

        return self._qdot_bar_all

    def erg_per_phot(self, Emin, Emax):
        return self.eV_per_phot(Emin, Emax) * erg_per_ev

    def eV_per_phot(self, Emin, Emax):
        """
        Compute the average energy per photon (in eV) in some band.
        """

        i1 = lambda E: self.get_spectrum(E)
        i2 = lambda E: self.get_spectrum(E) / E

        # Must convert units
        final = quad(i1, Emin, Emax, points=self.sharp_points, limit=50)[0] \
              / quad(i2, Emin, Emax, points=self.sharp_points, limit=50)[0]

        return final

    @property
    def sigma_bar(self):
        """
        Frequency averaged cross section (single bandpass).
        """
        if not hasattr(self, '_sigma_bar_all'):
            self._sigma_bar_all = np.zeros_like(self.grid.zeros_absorbers)
            for i, absorber in enumerate(self.grid.absorbers):
                integrand = lambda x: self.get_spectrum(x) \
                    * self.grid.bf_cross_sections[absorber](x) / x

                self._sigma_bar_all[i] = self.Lbol \
                    * quad(integrand, self.grid.ioniz_thresholds[absorber],
                      self.Emax, points=self.sharp_points)[0] / self.qdot_bar[i] / erg_per_ev

        return self._sigma_bar_all

    @property
    def sigma_tilde(self):
        if not hasattr(self, '_sigma_tilde_all'):
            self._sigma_tilde_all = np.zeros_like(self.grid.zeros_absorbers)
            for i, absorber in enumerate(self.grid.absorbers):
                integrand = lambda x: self.get_spectrum(x) \
                    * self.grid.bf_cross_sections[absorber](x)
                self._sigma_tilde_all[i] = quad(integrand,
                    self.grid.ioniz_thresholds[absorber], self.Emax,
                    points=self.sharp_points)[0] \
                    / self.fLbol_ionizing[i]

        return self._sigma_tilde_all

    @property
    def fLbol_ionizing(self, absorber=0):
        """
        Fraction of bolometric luminosity emitted above all ionization
        thresholds.
        """
        if not hasattr(self, '_fLbol_ioniz_all'):
            self._fLbol_ioniz_all = np.zeros_like(self.grid.zeros_absorbers)
            for i, absorber in enumerate(self.grid.absorbers):
                self._fLbol_ioniz_all[i] = quad(self.get_spectrum,
                    self.grid.ioniz_thresholds[absorber], self.Emax,
                    points=self.sharp_points)[0]

        return self._fLbol_ioniz_all

    @property
    def Gamma_bar(self):
        """
        Return ionization rate (as a function of radius) assuming optical
        depth to cells and of cells is small.
        """
        if not hasattr(self, '_Gamma_bar_all'):
            self._Gamma_bar_all = \
                np.zeros([self.grid.dims, self.grid.N_absorbers])
            for i, absorber in enumerate(self.grid.absorbers):
                self._Gamma_bar_all[..., i] = self.Lbol * self.sigma_bar[i] \
                    * self.fLbol_ionizing[i] / 4. / np.pi / self.grid.r_mid**2 \
                    / self.hnu_bar[i] / erg_per_ev

        return self._Gamma_bar_all

    @property
    def gamma_bar(self):
        """
        Return ionization rate (as a function of radius) assuming optical
        depth to cells and of cells is small.
        """
        if not hasattr(self, '_gamma_bar_all'):
            self._gamma_bar_all = \
                np.zeros([self.grid.dims, self.grid.N_absorbers,
                    self.grid.N_absorbers])

            if not self.pf['secondary_ionization']:
                return self._gamma_bar_all

            for i, absorber in enumerate(self.grid.absorbers):
                for j, otherabsorber in enumerate(self.grid.absorbers):
                    self._gamma_bar_all[..., i, j] = self.Gamma_bar[j] \
                        * (self.hnu_bar[j] * self.sigma_tilde[j] \
                        /  self.hnu_bar[i] / self.sigma_bar[j] \
                        - self.grid.ioniz_thresholds[otherabsorber] \
                        / self.grid.ioniz_thresholds[absorber])

        return self._gamma_bar_all

    @property
    def Heat_bar(self):
        """
        Return ionization rate (as a function of radius) assuming optical
        depth to cells and of cells is small.
        """
        if not hasattr(self, '_Heat_bar_all'):
            self._Heat_bar_all = \
                np.zeros([self.grid.dims, self.grid.N_absorbers])
            for i, absorber in enumerate(self.grid.absorbers):
                self._Heat_bar_all[..., i] = self.Gamma_bar[..., i] \
                    * erg_per_ev * (self.hnu_bar[i] * self.sigma_tilde[i] \
                    / self.sigma_bar[i] - self.grid.ioniz_thresholds[absorber])

        return self._Heat_bar_all

    def IonizingPhotonLuminosity(self, t=0, bin=None):
        """
        Return Qdot (photons / s) for this source at energy E.
        """

        if self.pf['source_type'] in [0, 1, 2]:
            return self.Qdot[bin]
        else:
            # Currently only BHs have a time-varying bolometric luminosity
            return self.BolometricLuminosity(t) * self.LE[bin] / self.E[bin] / erg_per_ev

    #def _Intensity(self, E, i, Type, t=0, absorb=True):
    #    """
    #    Return quantity *proportional* to fraction of bolometric luminosity emitted
    #    at photon energy E.  Normalization handled separately.
    #    """
    #
    #    Lnu = self.src._Intensity(E, i, Type, t=t)
    #
    #    # Apply absorbing column
    #    if self.SpectrumPars['logN'][i] > 0 and absorb:
    #        return Lnu * np.exp(-10.**self.SpectrumPars['logN'][i] \
    #            * (sigma_E(E, 0) + y * sigma_E(E, 1)))
    #    else:
    #        return Lnu
    #

    def get_spectrum(self, x, t=0.0, units='eV'):
        r"""
        Return fraction of bolometric luminosity emitted at energy E.

        Elsewhere denoted as :math:`I_{\nu}`, normalized such that
        :math:`\int I_{\nu} d\nu = 1`

        Parameters
        ----------
        x: float
            Emission energy in `units`.
        t: float
            Time in seconds since source turned on.
        i: int
            Index of component to include. If None, includes contribution
            from all components.

        Returns
        -------
        Fraction of bolometric luminosity emitted at E in units of
        eV\ :sup:`-1`\.

        """

        E = self.get_ev_from_x(x, units=units)

        if self.pf['source_Ekill'] is not None:
            if self.pf['source_Ekill'][0] <= E <= self.pf['source_Ekill'][1]:
                return 0.0

        return self._normL * self._Intensity(E, t=t)

    def BolometricLuminosity(self, t=0.0, M=None):
        """
        Returns the bolometric luminosity of a source in units of erg/s.
        For accreting black holes, the bolometric luminosity will increase
        with time, hence the optional 't' and 'M' arguments.
        """

        if self._name == 'bh':
            return self.Luminosity(t, M)
        else:
            return self.Luminosity(t)

    def _FrequencyAveragedBin(self, absorber='h_1', Emin=None, Emax=None,
        energy_weighted=False, t=0):
        """
        Bolometric luminosity / number of ionizing photons in spectrum in bandpass
        spanning interval (Emin, Emax). Returns mean photon energy and number of
        ionizing photons in band.
        """

        if Emin is None:
            Emin = max(self.grid.ioniz_thresholds[absorber], self.Emin)
        if Emax is None:
            Emax = self.Emax

        if energy_weighted:
            f = lambda x: x
        else:
            f = lambda x: 1.0

        L = self.Lbol * quad(lambda x: self.get_spectrum(x) * f(x), Emin, Emax,
            points=self.sharp_points)[0]
        Q = self.Lbol * quad(lambda x: self.get_spectrum(x) * f(x) / x, Emin,
            Emax, points=self.sharp_points)[0] / erg_per_ev

        return L / Q / erg_per_ev, Q

    def dump(self, fn, E=None, clobber=False):
        """
        Write SED out to file.

        Parameters
        ----------
        fn : str
            Filename, suffix determines type. If 'hdf5' or 'h5' will write
            to HDF5 file, otherwise, to ASCII.
        E : np.ndarray
            Array of photon energies at which to sample SED. Units = eV.

        """

        if os.path.exists(fn) and (clobber == False):
            raise OSError('{!s} exists!'.format(fn))

        if re.search('.hdf5', fn) or re.search('.h5', fn):
            out = 'hdf5'
        else:
            out = 'ascii'

        LE = list(map(self.get_spectrum, E))

        if out == 'hdf5':
            f = h5py.File(fn, 'w')
            f.create_dataset('E', data=E)
            f.create_dataset('LE', data=LE)
            f.close()
        else:
            f = open(fn, 'w')
            print("# E     LE", file=f)
            for i, nrg in enumerate(E):
                print("{0:.8e} {1:.8e}".format(nrg, LE[i]), file=f)
            f.close()

        print("Wrote {!s}.".format(fn))

    def sed_name(self, i=0):
        """
        Return name of output file based on SED properties.
        """

        name = ('{0!s}_logM_{1:.2g}_Gamma_{2:.3g}_fsc_{3:.3g}_' +\
            'logE_{4:.2g}-{5:.2g}').format(self.SpectrumPars['type'][i],\
            np.log10(self.src.M0), self.src.spec_pars['alpha'][i],
            self.src.spec_pars['fsc'][i], np.log10(self.Emin), np.log10(self.Emax))

        return name
