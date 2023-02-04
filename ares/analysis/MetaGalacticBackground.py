"""

AnalyzeXray.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Tue Sep 18 13:36:56 2012

Description:

"""

import numpy as np
from ..util import labels
from ..util.Pickling import read_pickle_file
import matplotlib.pyplot as pl
from scipy.integrate import trapz
from ..util.ReadData import flatten_energies
from ..physics.Constants import erg_per_ev, J21_num, h_P, c, E_LL, E_LyA, \
    sqdeg_per_std
try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

class MetaGalacticBackground(object):
    def __init__(self, data=None, **kwargs):
        """
        Initialize analysis object.

        Parameters
        ----------
        data : dict, str
            Either a dictionary containing the entire history or the prefix
            of the files containing the history/parameters.

        """

        if data is None:
            return
        elif type(data) == dict:
            self.pf = SetAllDefaults()
            self.history = data.copy()
        elif isinstance(data, basestring):
            self.prefix = data

        self.kwargs = kwargs

    @property
    def fluxes(self):
        if not hasattr(self, '_fluxes'):
            self._redshifts_fl, self._energies_fl, self._fluxes = \
                self._load_data('{!s}.fluxes.pkl'.format(self.prefix))
        return self._redshifts_fl, self._energies_fl, self._fluxes

    @property
    def emissivities(self):
        if not hasattr(self, '_emissivities'):
            self._redshifts_em, self._energies_em, self._emissivities = \
                self._load_data('{!s}.emissivities.pkl'.format(self.prefix))

        return self._redshifts_em, self._energies_em, self._emissivities

    def flat_flux(self, popid=0):
        return self._flatten_data(popid)

    def flat_emissivity(self, popid=0):
        return self._flatten_data(popid, True)

    def _flatten_data(self, popid=0, emissivity=False):
        """
        Re-organize background fluxes to a more easily-analyzable shape.

        Parameters
        ----------
        popid : int
            Population ID number.
        emissivity : bool
            If True, return emissivities. If False, return fluxes.
        """

        if not emissivity:
            z, E, data = self.fluxes
            Eflat = flatten_energies(E[popid])

            return z[popid], Eflat, data[popid]

        _z, _E, _data = self.emissivities
        E = _E[popid]
        z = _z[popid]
        Eflat = flatten_energies(E)
        flat = np.zeros([z.size, Eflat.size])
        data = _data[popid]

        k1 = 0
        k2 = 0

        # Each population's spectrum is broken down into bands, which are
        # defined by their relation to ionization thresholds and Ly-n series'
        for i, band in enumerate(E):

            # OK, a few things can happen next.
            # 1. Easy case: a contiguous chunk of spectrum, meaning there
            # are no sawtooth sections or any such thing.
            # 2. Hard case: a chunk of spectrum sub-divided into many
            # sub-chunks, which will in general not be the same shape.

            if type(data[i]) is list:

                # Create a flattened array
                for j, element in enumerate(band):
                    len_el = len(element)
                    k2 += len_el
                    flat[:,k1:k2] = data[i][j]
                    k1 += len_el

            else:
                len_band = len(band)
                k2 += len_band
                flat[:,k1:k2] = data[i]
                k1 += len_band

        # Why do I keep fluxes and emissivities in different units?
        # I'd hope there is a decent reason...
        flat /= (Eflat * erg_per_ev)

        return z, Eflat, flat

    def _load_data(self, fn):
        (redshifts, energies, data) =\
            read_pickle_file(fn, nloads=1, verbose=False)

        try:
            parameters_fn = '{!s}.parameters.pkl'.format(self.prefix)
            self.pf = read_pickle_file(parameters_fn, nloads=1, verbose=False)

        # The import error is really meant to catch pickling errors
        except (AttributeError, ImportError):
            self.pf = {"final_redshift": 5., "initial_redshift": 100.}
            print('Error loading {!s}.parameters.pkl.'.format(data))

        return redshifts, energies, data

    def _obs_xrb(self, fit='moretti2012'):
        """
        Operations on the best fit to the CXRB from Moretti et al. 2009 from 2keV-2MeV.

        Energy units are in keV.
        Flux units are keV^2/cm^2/s/keV/sr

        Convert energies to eV throughout, change default flux units
        Include results from Markevich & Hickox

        """

        self.fit = fit

        # Fit parameters from Moretti et al. 2009 (Model = 2SJPL, Table 2)
        if fit in ['moretti2009', 'moretti2009+2SJPL']:
            self.C = 0.109 / sqdeg_per_std / 1e3    # now photons / s / cm^2 / deg^2 / eV
            self.C_err = 0.003 / sqdeg_per_std / 1e3
            self.Gamma1 = 1.4
            self.Gamma1_err = 0.02
            self.Gamma2 = 2.88
            self.Gamma2_err = 0.05
            self.EB = 29.0 * 1e3
            self.EB_err = 0.5 * 1e3
            self.integrated_2_10_kev_flux = 2.21e-11
            self.sigma_integrated_2_10_kev_flux = 0.07e-11
        elif fit == 'moretti2009+PL':
            self.C = 3.68e-3
        elif fit == 'moretti2012':
            self.unresolved_2_10_kev_flux = 5e-12
            self.sigma_unresolved_2_10_kev_flux = 1.77e-12

        # Some defaults
        self.E = np.logspace(2, 5, 100)    # eV

    def broadband_flux(self, z):
        """
        Sum over populations and stitch together different bands to see the
        meta-galactic background flux over a broad range of energies.

        .. note:: This assumes you've already run the calculation to
            completion.

        Parameters
        ----------
        z : int, float
            Redshift of interest.

        """

        fluxes_by_band = {}
        energies_by_band = {}
        for i, source in enumerate(self.field.sources):
            for band in self.field.bands:
                if band not in fluxes_by_band:
                    fluxes_by_band[band] = \
                        np.zeros_like(self.field.energies[i])
                    energies_by_band[band] = \
                        list(self.field.energies[i])

                j = np.argmin(np.abs(z - self.field.redshifts[i][-1::-1]))
                if self.field.redshifts[i][-1::-1][j] > z:
                    j1 = j - 1
                else:
                    j1 = j

                j2 = j1 + 1

                flux = np.zeros_like(self.field.energies[i])
                for k, nrg in enumerate(self.field.energies[i]):
                    flux[k] += \
                        np.interp(z, self.field.redshifts[i][-1::-1][j1:j2+1],
                        self.history[i][j1:j2+1,k])

                fluxes_by_band[band] += flux

        # Now, stitch everything together
        E = []; F = []
        for band in fluxes_by_band:
            E.extend(energies_by_band[band])
            F.extend(fluxes_by_band[band])

        return np.array(E), np.array(F)



    def ResolvedFlux(self, E=None, perturb=False):
        """
        Return total CXRB flux (resolved + unresolved) in erg / s / cm^2 / deg^2.
        Plotting this should reproduce Figure 11 in Moretti et al. 2009 (if
        fit == 1) modulo the units.
        """

        if E is None:
            E = self.E

        # Randomly vary model parameters assuming their 1-sigma (Gaussian) errors
        if perturb:
            Cerr = np.random.normal(scale = self.C_err)
            EBerr = np.random.normal(scale = self.EB_err)
            Gamma1err = np.random.normal(scale = self.Gamma1_err)
            Gamma2err = np.random.normal(scale = self.Gamma2_err)
        else:
            Cerr = EBerr = Gamma1err = Gamma2err = 0.0

        # Return SWIFT-BAT measured CXRB flux at energy E (keV).
        if self.fit == 'moretti2009':
            self.integrated_2_10_kev_flux = 2.21e-11
            self.sigma_integrated_2_10_kev_flux = 0.07e-11

            flux = E**2 * (self.C + Cerr) \
                / ((E / (self.EB + EBerr))**(self.Gamma1 + Gamma1err) \
                + (E / (self.EB + EBerr))**(self.Gamma2 + Gamma2err))

            return flux * erg_per_ev # erg / s / cm^2 / deg^2

    def IntegratedFlux(self, Emin=2e3, Emax=1e4, Nbins=1e3, perturb=False):
        """
        Integrated flux in [Emin, Emax] (eV) band.
        """

        E = np.logspace(np.log10(Emin), np.log10(Emax), Nbins)
        F = self.ResolvedFlux(E, perturb=perturb) / E

        return trapz(F, E)  # erg / s / cm^2 / deg^2

    def PlotIntegratedFlux(self, E, **kwargs):
        return self.PlotSpectrum(E, vs_redshift=True, **kwargs)

    def PlotMonochromaticFlux(self, E, **kwargs):
        return self.PlotSpectrum(E, vs_redshift=True, **kwargs)

    def PlotSpectrum(self, x, vs_redshift=False, ax=None, fig=1,
        xaxis='energy', overplot_edges=False, band=None, units='J21', popid=0,
        emissivity=False, **kwargs): # pragma: no cover
        """
        Plot meta-galactic background intensity at a single redshift.

        Parameters
        ----------
        z : int, float
            Redshift of interest

        """

        if ax is None:
            gotax = False
            fig = pl.figure(fig)
            ax = fig.add_subplot(111)
        else:
            gotax = True

        # Read in the history
        zarr, Earr, flux = self._flatten_data(popid=popid,
            emissivity=emissivity)

        if vs_redshift:
            if x is None:
                i_z = -1
            else:
                if (x < Earr.min()) or (x > Earr.max()):
                    raise ValueError("Requested E lies below provied range.")

                i_z = np.argmin(np.abs(x - Earr))

            # Convert units (native unit is photons, not energy)
            f = flux[:,i_z] * Earr[i_z] * erg_per_ev
        else:
            if x is None:
                i_z = -1
            else:
                if x < zarr.min():
                    raise ValueError("Requested z lies below provied range.")

                i_z = np.argmin(np.abs(x - zarr))

            # Convert units (native unit is photons, not energy)
            f = flux[i_z] * Earr * erg_per_ev

        if units.lower() == 'j21':
            f /= 1e-21
        elif units.lower() == 'nuFnu':
            f *= Earr * erg_per_ev * ev_per_hz / 1e3

        if vs_redshift:
            xarr = zarr
        elif xaxis == 'energy':
            xarr = Earr
        else:
            xarr = h_P * c * 1e8 / (Earr * erg_per_ev)

        ax.plot(xarr, f, **kwargs)

        if vs_redshift:
            ax.set_xlabel(labels['z'])
        elif xaxis == 'energy' and not gotax:
            ax.set_xlabel(labels['E'])
        elif not gotax:
            ax.set_xlabel(labels['lambda_AA'])

        # Add HI and HeII Lyman-alpha and Lyman-limit
        if overplot_edges and not gotax:
            if xaxis == 'energy':
                for nrg in [E_LyA, E_LL, 4*E_LyA, 4*E_LL]:
                    ax.plot([nrg]*2, ax.get_ylim(), color='k', ls=':')
            else:
                for nrg in [E_LyA, E_LL, 4*E_LyA, 4*E_LL]:
                    ax.plot([h_P * c / (nrg * erg_per_ev)]*2, ax.get_ylim(),
                        color='k', ls=':')

        if units.lower() == 'j21':
            ax.set_ylabel(r'$J_{\nu} / J_{21}$')
        elif units.lower() == 'nuFnu':
            ax.set_ylabel(r'$\mathrm{keV} \ \mathrm{cm}^{-2} \ \mathrm{s}^{-1} \ \mathrm{sr}^{-1}$')

        pl.draw()

        if xaxis == 'energy' and not gotax:
            self.twinax = self.add_wavelength_axis(ax)
        elif not gotax:
            self.twinax = self.add_energy_axis(ax)

        return ax

    def add_wavelength_axis(self, ax):
        """
        Take plot with redshift on x-axis and add top axis with corresponding
        (observed) 21-cm frequency.

        Parameters
        ----------
        ax : matplotlib.axes.AxesSubplot instance
        """

        return

        l = np.arange(10, 4000, 20)
        l_minor = np.arange(30, 230, 20)
        znu = nu_0_mhz / np.array(nu) - 1.
        znu_minor = nu_0_mhz / np.array(nu_minor) - 1.

        ax_freq = ax.twiny()
        ax_freq.set_xlabel(labels['nu'])
        ax_freq.set_xticks(znu)
        ax_freq.set_xticks(znu_minor, minor=True)
        ax_freq.set_xlim(ax.get_xlim())

        freq_labels = list(map(str, nu))

        # A bit hack-y
        for i, label in enumerate(freq_labels):
            if label in ['140', '180']:
                freq_labels[i] = ''

            if float(label) >= 200:
                freq_labels[i] = ''

        ax_freq.set_xticklabels(freq_labels)

        pl.draw()

        return ax_freq

    def add_energy_axis(self, ax):
        """
        Add rest-frame photon energies on top of plot, assumed to be in
        wavelength units on the bottom x-axis.

        Parameters
        ----------
        ax : matplotlib.axes.AxesSubplot instance

        """

        return

        E = 10**np.arange(0., 5.)[-1::-1]
        #Eminor = np.array([25, 75, 250, 750])[-1::-1]
        wave = 1e8 * h_P * c / (E * erg_per_ev)
        #wave_minor = 1e8 * h_P * c / (Eminor * erg_per_ev)

        #wave = ax.get_xticks()

        ax_nrg = ax.twiny()
        ax_nrg.set_xticks(wave)
        #ax_nrg.set_xscale('log')
        #ax_nrg.xaxis.tick_top()
        ax_nrg.set_xlabel(labels['E'])

        #ax_nrg.set_xticks(wave_minor, minor=True)

        #ax_nrg.set_xticklabels(list(map(str, E)))
        ax_nrg.set_xlim(ax.get_xlim())
        ax_nrg.set_xscale('log')

        pl.draw()

        return ax_nrg
