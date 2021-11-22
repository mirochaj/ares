"""
RaySegment.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2011-06-17.

Description: Functions to calculate various quantities from our rt1d datasets.

"""

import os, re
import numpy as np
from ..util import labels
from math import floor, ceil
import matplotlib.pyplot as pl
from ..static.Grid import Grid
from ..physics.Constants import *
from ..util.SetDefaultParameterValues import *
from .MultiPhaseMedium import HistoryContainer

try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

try:
    import h5py
except ImportError:
    pass

linestyles = ['-', '--', ':', '-.']

class RaySegment(object):
    def __init__(self, data=None, **kwargs):
        """
        Initialize analysis object for RaySegment calculations.

        Parameters
        ----------
        data : instance, np.ndarray
            Dataset to analyze.

        """

        if data is None:
            return

        # Not sure why we need the isinstance here but not in
        # MultiPhaseMedium
        elif type(data) == dict:
            self.pf = SetAllDefaults()
            self.history = data.copy()

        # Read output of a simulation from disk
        elif isinstance(data, basestring):
            self.prefix = data
            self._load_data(data)

        self.kwargs = kwargs

        # Read in
        #if isinstance(data, simRS):
        #    self.sim = data
        #    self.pf = data.pf
        #    self.data = data.history
        #    self.grid = data.parcel.grid
        #
        ## Load contents of hdf5 file
        #elif isinstance(data, basestring):
        #    f = h5py.File(data, 'r')
        #
        #    self.pf = {}
        #    for key in f['parameters']:
        #        self.pf[key] = f['parameters'][key].value
        #
        #    self.data = {}
        #    for key in f.keys():
        #        if not f[key].attrs.get('is_data'):
        #            continue
        #
        #        if key == 'parameters':
        #            continue
        #
        #        dd = key#int(key.strip('dd'))
        #        self.data[dd] = {}
        #        for element in f[key]:
        #            self.data[dd][element] = f[key][element].value
        #
        #    f.close()
        #
        #    self.grid = Grid(dims=self.pf['grid_cells'],
        #        length_units=self.pf['length_units'],
        #        start_radius=self.pf['start_radius'],
        #        approx_Salpha=self.pf['approx_Salpha'],
        #        approx_lwb=self.pf['approx_lwb'])
        #
        #    self.grid.set_ics(self.data['dd0000'])
        #    self.grid.set_chemistry(self.pf['include_He'])
        #    self.grid.set_density(self.data['dd0000']['rho'])
        #
        ## Read contents from CheckPoints class instance
        #else:
        #    self.checkpoints = checkpoints
        #    self.grid = checkpoints.grid
        #    self.pf = checkpoints.pf
        #    self.data = checkpoints.data

    @property
    def history(self):
        return self._history

    @history.setter
    def history(self, value):
        if not hasattr(self, '_history'):
            self._history = HistoryContainer(pf=self.pf)
            self._history.add_data(value)

        return self._history

    @property
    def tcode(self):
        if not hasattr(self, '_tcode'):
            self._tcode = self.history['t'] / self.pf['time_units']

        return self._tcode

    def get_snapshot(self, t):
        """
        Grab dataset associated with time `t`.

        Parameters
        ----------
        t : int, float
            Time corresponding to dataset you want, in code units.

        Returns
        -------
        Dictionary containing data for this snapshot only.

        """

        loc = np.argmin(np.abs(t - self.tcode))

        to_return = {}
        for key in self.history:
            if len(self.history[key].shape) == 1:
                continue

            to_return[key] = self.history[key][loc,:]

        return to_return

    def get_evolution(self, cell):
        """
        Grab time-series data for a particular cell.

        Parameters
        ----------
        cell : int
            Grid `cell` you're interested in.

        Returns
        -------
        Dictionary containing data for this cell.

        """

        to_return = {}
        for key in self.history:
            if len(self.history[key].shape) == 1:
                continue

            to_return[key] = self.history[key][:,cell]

        return to_return

    def EmergentSpectrum(self, t, cell=-1, norm=False, logE=False):
        """
        Compute emergent spectrum at time t (Myr) and given cell in grid. By
        default, cell=-1, i.e., compute the spectrum that emerges from the grid.
        If norm=True, do not apply geometrical dilution.

        Returns
        -------
        E, Fin, Fout

        """

        if not hasattr(self, 'rs'):
            raise ValueError('RadiationSource class instance required.')

        if logE:
            E = np.logspace(np.log10(self.rs.Emin), np.log10(self.rs.Emax))
        else:
            E = np.linspace(self.rs.Emin, self.rs.Emax)

        F = np.array(list(map(self.rs.Spectrum, E)))

        for dd in self.history:
            if self.history[dd]['time'] / s_per_myr != t:
                continue

            N, logN, Nc = self.grid.ColumnDensity(self.history[dd])

            Ntot = {}
            for absorber in self.grid.absorbers:
                Ntot[absorber] = N[absorber][cell]

            tau = self.rs.tab.SpecificOpticalDepth(E, Ntot)

            break

        out = F * np.exp(-tau)
        if not norm:
            out *= self.rs.BolometricLuminosity(t * s_per_myr) \
                / (4. * np.pi * self.grid.r[cell]**2)

        return E, F, out

    def StromgrenSphere(self, t, sol=0, T0=None):
        """
        Classical analytic solution for expansion of an HII region in an
        isothermal medium.  Given the time in seconds, will return the I-front
        radius in centimeters.

        Future: sol = 1 will be the better "analytic" solution.
        """

        # Stuff for analytic solution
        if sol == 0:
            if T0 is not None:
                T = T0
            else:
                T = self.history['Tk'][0,0]

            n_H = self.grid.n_H[0]
            self.Qdot = self.pf['source_qdot']
            self.alpha_HII = 2.6e-13 * (T / 1.e4)**-0.85
            self.trec = 1. / self.alpha_HII / self.history['h_1'][0,0] / n_H # s
            self.rstrom = (3. * self.Qdot \
                    / 4. / np.pi / self.alpha_HII / n_H**2)**(1. / 3.)  # cm
        else:
            raise NotImplementedError('help')

        return self.rstrom * (1. - np.exp(-t / self.trec))**(1. / 3.) \
            + self.pf['start_radius']

    def LocateIonizationFront(self, t, species='h_1'):
        """
        Find the position of the ionization front.

        Parameters
        ----------
        t : int, float
            Time in code units.
        species : str
            For example, h_1, he_1, he_2.

        """

        data = self.get_snapshot(t)
        return np.interp(0.5, data[species], self.grid.r_mid)

    def ComputeIonizationFrontEvolution(self, T0=None, xmin=0.1, xmax=0.9):
        """
        Find the position of the I-front at all times, and compute value of
        analytic solution.
        """

        # First locate I-front for all data dumps and compute analytic solution
        self.t = []
        self.rIF = []
        self.drIF, self.r1_IF, self.r2_IF = [], [], []
        self.ranl = []
        for i, t in enumerate(self.history['t']):
            if t == 0:
                continue

            tcode = t / self.pf['time_units']

            data = self.get_snapshot(tcode)



            self.t.append(t)
            self.rIF.append(self.LocateIonizationFront(tcode) / cm_per_kpc)
            self.ranl.append(self.StromgrenSphere(t, T0=T0) / cm_per_kpc)

            x1 = np.interp(1.-xmax, data['h_1'], self.grid.r_mid)
            x2 = np.interp(1.-xmin, data['h_1'], self.grid.r_mid)
            self.drIF.append(x2-x1)
            self.r1_IF.append(x1)
            self.r2_IF.append(x2)

        self.t = np.array(self.t)
        self.rIF = np.array(self.rIF)
        self.ranl = np.array(self.ranl)
        self.drIF = np.array(self.drIF)
        self.r1_IF = np.array(self.r1_IF)
        self.r2_IF = np.array(self.r2_IF)

    def PlotIonizationFrontEvolution(self, fig=1, axes=None, anl=True, T0=1e4,
        color='k', ls='--', label=None, plot_error=True, plot_solution=True): # pragma: no cover
        """
        Compute analytic and numerical I-front radii vs. time and plot.
        """

        self.ComputeIonizationFrontEvolution(T0=T0)

        had_axes = False
        if axes is not None:
            had_axes = True
        else:
            fig, axes = pl.subplots(2, 1, num=fig)

        if anl:
            axes[1].plot(self.t / self.trec, self.ranl, ls='-', color='k')

        if plot_solution:
            axes[1].plot(self.t / self.trec, self.rIF,
                color = color, ls = ls)
            axes[1].set_xlim(0, max(self.t / self.trec))
            axes[1].set_ylim(0, 1.1 * max(max(self.rIF), max(self.ranl)))
            axes[1].set_ylabel(r'$r \ (\mathrm{kpc})$')

        maxt = max(self.t / self.trec)

        if plot_error:
            axes[0].plot(self.t / self.trec, self.rIF / self.ranl,
                color=color, ls=ls, label=label)
            axes[0].set_xlim(0, maxt)
            axes[0].set_ylim(0.8, 1.04)

            if not hadmp:
                #axes[0].set_yticks(np.arange(0.94, 1.04, 0.02))
                axes[0].set_xlabel(r'$t / t_{\mathrm{rec}}$')
                axes[0].set_ylabel(r'$r_{\mathrm{num}} / r_{\mathrm{anl}}$')
                axes[1].set_xticks(np.arange(0, round(maxt), 0.25))
                axes[0].set_xticks(np.arange(0, round(maxt), 0.25))
                axes[1].set_xticklabels(np.arange(0, round(maxt), 0.25))
                axes[0].set_xticklabels(np.arange(0, round(maxt), 0.25))

        pl.draw()

        return axes

    def PlotRadialProfile(self, field, t=[1, 10, 100], ax=None, fig=1,
        **kwargs): # pragma: no cover
        """
        Plot radial profile of given field.

        Parameters
        ----------
        field : str
            Field you'd like to plot a radial profile of, e.g., `h_1`.
        t : int, float, list
            Time(s) at which to plot radial profile.
        """

        if ax is None:
            gotax = False
            fig = pl.figure(fig)
            ax = fig.add_subplot(111)
        else:
            gotax = True

        for tcode in t:
            data = self.get_snapshot(tcode)
            ax.plot(self.grid.r_mid / cm_per_kpc, data[field], **kwargs)

        # Don't add more labels
        if gotax:
            pl.draw()
            return ax

        # Label stuff!
        ax.set_xlabel(r'$r \ (\mathrm{kpc})$')

        if field in labels:
            ax.set_ylabel(labels[field])
        else:
            ax.set_ylabel(field)

        pl.draw()
        return ax

    def get_cell_evolution(self, cell=0, field='h_1', redshift=False):
        """
        Return time or redshift evolution of a given quantity in given cell.
        """

        if field not in self.history.keys():
            raise KeyError('No field {!s} in dataset.'.format(field))

        if redshift:
            raise NotImplemented('sorry about redshift=True!')

        return self.history['t'], self.history[field][:,cell]

    def PlotIonizationRate(self, t=1, absorber='h_1', color='k', ls='-',
        legend=True, plot_recomb=False, total_only=False, src=0,
        ax=None): # pragma: no cover
        """
        Plot ionization rate.

        Parameters
        ----------
        total_only : bool
            If True, will only plot total ionization rate. If False, will
            separate out into photo-ionization, secondary ionization,
            and collisional ionization.

        """

        if type(t) is not list:
            t = [t]

        if ax is None:
            ax = pl.subplot(111)

        i = self.grid.absorbers.index(absorber)
        for dd in self.history.keys():
            if self.history[dd]['time'] / self.pf['time_units'] not in t:
                continue

            ne = self.history[dd]['de']
            nabs = self.history[dd][absorber] * self.grid.x_to_n[absorber]
            nion = self.history[dd]['h_2'] * self.grid.x_to_n[absorber]

            if 'Gamma_0' in self.history[dd].keys():
                Gamma = self.history[dd]['Gamma_{}'.format(src)][...,i] * nabs

                gamma = 0.0
                for j, donor in enumerate(self.grid.absorbers):
                    gamma += self.history[dd]['gamma_{}'.format(src)][...,i,j] * \
                        self.history[dd][donor] * self.grid.x_to_n[donor]

            else:
                Gamma = self.history[dd]['Gamma'][...,i] * nabs

                gamma = 0.0
                for j, donor in enumerate(self.grid.absorbers):
                    gamma += self.history[dd]['gamma'][...,i,j] * \
                        self.history[dd][donor] * self.grid.x_to_n[donor]

            if 'Beta' in self.history[dd]:
                Beta = self.history[dd]['Beta'][...,i] * nabs * ne
            else:
                Beta = 0

            ion = Gamma + Beta + gamma # Total ionization rate

            # Recombinations
            alpha = self.history[dd]['alpha'][...,i] * nion * ne
            xi = self.history[dd]['xi'][...,i] * nion * ne
            recomb = alpha + xi

            ax.loglog(self.grid.r_mid / cm_per_kpc, ion,
                color=color, ls=ls, label='Total')

            if not total_only:
                ax.loglog(self.grid.r_mid / cm_per_kpc, Gamma,
                    color=color, ls='--', label=r'$\Gamma$')
                ax.loglog(self.grid.r_mid / cm_per_kpc, gamma,
                    color=color, ls=':', label=r'$\gamma$')
                ax.loglog(self.grid.r_mid / cm_per_kpc, Beta,
                    color=color, ls='-.', label=r'$\beta$')

            if plot_recomb:
                ax.loglog(self.grid.r_mid / cm_per_kpc, recomb,
                    color = 'b', ls = '-', label = 'Recomb.')

                if not total_only:
                    ax.loglog(self.grid.r_mid / cm_per_kpc, alpha,
                        color = 'b', ls = '--', label = r'$\alpha$')
                    ax.loglog(self.grid.r_mid / cm_per_kpc, xi,
                        color = 'b', ls = ':', label = r'$\xi$')

        ax.set_xlabel(r'$r \ (\mathrm{kpc})$')
        ax.set_ylabel(r'Ionization Rate $(\mathrm{s}^{-1})$')
        ax.set_ylim(0.01 * 10**np.floor(np.log10(np.min(ion))),
            10**np.ceil(np.log10(np.max(ion))))

        if legend:
            ax.legend(frameon=False, ncol=2, loc='best')

        pl.draw()

        return ax

    def PlotHeatingRate(self, t=1, color='r', ls='-', legend=True, src=0,
        plot_cooling=False, label=None, ax=None): # pragma: no cover
        """
        Plot heating rate as a function of radius.

        Parameters
        ----------
        t : int, float
            Time (code units) at which to plot heating rate.
        plot_cooling : bool
            Plot total cooling rate vs. radius as well?

        """

        if type(t) is not list:
            t = [t]

        if ax is None:
            ax = pl.subplot(111)

        if label is None:
            heat_label = r'$\mathcal{H}_{\mathrm{tot}}$'
        else:
            heat_label = label

        ax = pl.subplot(111)
        for dd in self.history.keys():
            if self.history[dd]['time'] / self.pf['time_units'] not in t:
                continue

            ne = self.history[dd]['de']
            heat, zeta, eta, psi, omega, cool = np.zeros([6, self.grid.dims])
            for absorber in self.grid.absorbers:
                i = self.grid.absorbers.index(absorber)

                nabs = self.history[dd][absorber] * self.grid.x_to_n[absorber]
                nion = self.history[dd]['h_2'] * self.grid.x_to_n[absorber]

                # Photo-heating
                if 'Heat_0' in self.history[dd].keys():
                    heat = heat +\
                        self.history[dd]['Heat_{}'.format(src)][...,i] * nabs
                else:
                    heat = heat + self.history[dd]['Heat'][...,i] * nabs

                # Cooling
                zeta = zeta + self.history[dd]['zeta'][...,i] * nabs * ne # collisional ionization
                eta = eta + self.history[dd]['eta'][...,i] * nion * ne   # recombination
                psi = psi + self.history[dd]['psi'][...,i] * nabs * ne   # collisional excitation

                if absorber == 'he_2':
                    omega = self.history[dd]['omega'] * nion * ne # dielectric

            cool = (zeta + eta + psi + omega)
            #if self.pf['CosmologicalExpansion']:
            #    cool += self.history[dd].hubble * 3. * self.history[dd].T * k_B * self.history[dd].n_B

            mi = min(np.min(heat), np.min(cool))
            ma = max(np.max(heat), np.max(cool))

            ax.loglog(self.grid.r_mid / cm_per_kpc, heat,
                color = color, ls = ls, label = heat_label)

            if plot_cooling:
                ax.loglog(self.grid.r_mid / cm_per_kpc, cool,
                    color = 'b', ls = '-', label = r'$\mathcal{C}_{\mathrm{tot}}$')
                ax.loglog(self.grid.r_mid / cm_per_kpc, zeta,
                    color = 'g', ls = '--', label = r'$\zeta$')
                ax.loglog(self.grid.r_mid / cm_per_kpc, psi,
                    color = 'g', ls = ':', label = r'$\psi$')
                ax.loglog(self.grid.r_mid / cm_per_kpc, eta,
                    color = 'c', ls = '--', label = r'$\eta$')

                if 'he_2' in self.grid.absorbers:
                    ax.loglog(self.grid.r_mid / cm_per_kpc, omega,
                        color = 'c', ls = ':', label = r'$\omega_{\mathrm{HeII}}$')

                #if self.pf['CosmologicalExpansion']:
                #    self.ax.loglog(self.history[dd].r / cm_per_kpc,
                #        self.history[dd]['hubble'] * 3. * self.history[dd].T * k_B * self.history[dd].n_B,
                #        color = 'm', ls = '--', label = r'$H(z)$')

            if plot_cooling:
                ax_label = r'Heating & Cooling Rate $(\mathrm{erg/s/cm^3})$'
            else:
                ax_label = r'Heating Rate $(\mathrm{erg} \ \mathrm{s}^{-1} \ \mathrm{cm}^{-3})$'

        ax.set_xlabel(r'$r \ (\mathrm{kpc})$')
        ax.set_ylabel(ax_label)
        ax.set_ylim(0.001 * 10**np.floor(np.log10(mi)),
            10**np.ceil(np.log10(ma)))

        if legend:
            ax.legend(frameon=False, ncol=3, loc='best')

        pl.draw()

        self.heat = heat
        self.cool, self.zeta, self.eta, self.psi = (cool, zeta, eta, psi)

        return ax
