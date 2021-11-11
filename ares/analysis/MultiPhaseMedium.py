"""

MultiPhaseMedium.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Oct  4 12:59:46 2012

Description:

"""
import numpy as np
import re, scipy, os
from ..util import labels
import matplotlib.pyplot as pl
from ..util.Stats import get_nu
from ..util.Pickling import read_pickle_file
from scipy.misc import derivative
from ..physics.Constants import *
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
from ..physics import Cosmology, Hydrogen
from ..util.SetDefaultParameterValues import *
from mpl_toolkits.axes_grid1 import inset_locator
from .DerivedQuantities import DerivedQuantities as DQ
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

class HistoryContainer(dict):
    """
    A wrapper around DerivedQuantities.
    """
    def __init__(self, pf={}, cosm=None):
        self._data = {}
        self._dq = DQ(ModelSet=None, cosm=cosm, pf=pf)

    def add_data(self, data):
        self._dq._add_data(data)
        for key in data:
            self._data[key] = data[key]

    def keys(self):
        return self._data.keys()

    def __iter__(self):
        for key in self._data.keys():
            yield key

    def __contains__(self, name):
        return name in self._data.keys()

    def __getitem__(self, name):
        if name in self._data:
            return self._data[name]

        self._data[name] = self._dq[name]
        return self._data[name]

    def __setitem__(self, name, value):
        self._data[name] = value
        self._dq._data[name] = value

    def update(self, data):
        self._data.update(data)
        self._dq._data.update(data)

turning_points = ['D', 'C', 'B', 'A']

class MultiPhaseMedium(object):
    def __init__(self, data=None, suffix='history', **kwargs):
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

        # Read output of a simulation from disk
        elif isinstance(data, basestring):
            self.prefix = data
            self._load_data(data)

        self.kwargs = kwargs

    def _load_data(self, data):
        if os.path.exists('{!s}.history.pkl'.format(data)):
            history = self._load_pkl(data)
        else:
            history = self._load_txt(data, suffix)

        self._load_pf(data)
        self.history = history

    def _load_pf(self, data):
        try:
            self.pf = read_pickle_file('{!s}.parameters.pkl'.format(data),\
                nloads=1, verbose=False)
        # The import error is really meant to catch pickling errors
        except (AttributeError, ImportError):
            self.pf = {"final_redshift": 5., "initial_redshift": 100.}
            print('Error loading {!s}.parameters.pkl.'.format(data))

    def _load_txt(self, data, histsuffix):
        found = False
        for suffix in ['txt', 'dat']:
            fn = '{0!s}.history.{1!s}'.format(data, suffix)
            if os.path.exists(fn):
                found = True
                break

        if not found:
            raise IOError('Couldn\'t find file of form {!s}.history*'.format(\
                data))

        with open(fn, 'r') as f:
            cols = f.readline()[1:].split()

        data = np.loadtxt(fn, unpack=True)

        return {key:data[i] for i, key in enumerate(cols)}

    def _load_pkl(self, data):
        try:
            if not hasattr(self, '_suite'):
                self._suite = []
            fn = '{!s}.history.pkl'.format(data)
            loaded_chunks = read_pickle_file(fn, nloads=None, verbose=False)
            self._suite.extend(loaded_chunks)
            if len(loaded_chunks) == 0:
                raise IOError('Empty history ({!s}.history.pkl)'.format(data))
            else:
                history = self._suite[-1]
        except IOError:
            if re.search('pkl', data):
                history = read_pickle_file(data, nloads=1, verbose=False)
            else:
                import glob
                fns = glob.glob('./{!s}.history*'.format(data))
                if not fns:
                    raise IOError('No files with prefix {!s}.'.format(data))
                else:
                    fn = fns[0]

                f = open(fn, 'r')
                cols = f.readline().split()[1:]
                _data = np.loadtxt(f)

                history = {}
                for i, col in enumerate(cols):
                    history[col] = _data[:,i]
                f.close()

        return history

    @property
    def cosm(self):
        if not hasattr(self, '_cosm'):
            self._cosm = Cosmology(**self.pf)

        return self._cosm

    @property
    def hydr(self):
        if not hasattr(self, '_hydr'):
            self._hydr = Hydrogen(cosm=self.cosm, **self.pf)

        return self._hydr

    #@property
    #def data(self):
    #    if not hasattr(self, '_data'):
    #        if hasattr(self, 'history'):
    #            self._data = HistoryContainer(pf=self.pf)
    #            self._data.add_data(self.history)
    #
    #    return self._data

    @property
    def blobs(self):
        if not hasattr(self, '_blobs'):
            self._blobs =\
                read_pickle_file('{!s}.blobs.pkl'.format(self.prefix),\
                nloads=1, verbose=False)
        return self._blobs

    def close(self):
        pl.close('all')

    def draw(self):
        pl.draw()

    def show(self):
        pl.show()

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
    def history_asc(self):
        if not hasattr(self, '_history_asc'):
            if np.all(np.diff(self.history['z']) > 0):
                data_reorder = self.history.copy()
            else:
                data_reorder = {}
                for key in self.history.keys():
                    data_reorder[key] = np.array(self.history[key])[-1::-1]

            self._history_asc = data_reorder

        return self._history_asc

   #@property
   #def data_asc(self):
   #    if not hasattr(self, '_data_asc'):
   #        if np.all(np.diff(self.history['z']) > 0):
   #            data_reorder = self.history.copy()
   #        else:
   #            data_reorder = {}
   #            for key in self.history.keys():
   #                data_reorder[key] = np.array(self.history[key])[-1::-1]
   #
   #        self._data_asc = data_reorder
   #
   #    return self._data_asc

    @property
    def Trei(self):
        if not hasattr(self, '_Trei'):
            self._Trei = self.turning_points[1][0]
        return self._Trei

    def _initialize_interpolation(self, field):
        """ Set up scipy.interpolate.interp1d instances for all fields. """

        self.interp[field] = scipy.interpolate.interp1d(self.history_asc['z'],
            self.history_asc[field], kind='cubic')

    def field(self, z, field='igm_dTb'):
        """ Interpolate value of any field in self.history to redshift z. """

        if field not in self.interp:
            self._initialize_interpolation(field)

        return self.interp[field](z)

    def z_to_mhz(self, z):
        return nu_0_mhz / (1. + z)

    def zrei(self):
        """
        Find midpoint of reionization.
        """

        zz = self.history['z'][self.history['z'] < 40]
        xz = self.history['cgm_h_2'][self.history['z'] < 40]

        return np.interp(0.5, xz, zz)

    def excluded_regions(self, ax): # pragma: no cover
        """
        Overplot
        """

        ax.get_xlim()

        # Adiabatic cooling limit
        sim = simG21(tanh_model=True, tanh_T0=0.0,
            tanh_x0=0.0, tanh_J0=1e5, tanh_Jz0=120, initial_redshift=120)

        ax.fill_between(sim.history['nu'], sim.history['igm_dTb'],
            -500 * np.ones_like(sim.history['nu']), hatch='x', color='none',
            edgecolor='gray')

        # Saturated
        sim = simG21(tanh_model=True, tanh_x0=0.0,
            tanh_J0=1e5, tanh_Jz0=120, tanh_Tz0=120, tanh_T0=1e4,
            initial_redshift=120)

        ax.fill_between(sim.history['nu'], sim.history['igm_dTb'],
            80 * np.ones_like(sim.history['nu']), hatch='x', color='none',
            edgecolor='gray')

        ax.set_xlim()

        pl.draw()

    def tau_CMB(self, include_He=True, z_HeII_EoR=3.):
        """
        Compute CMB optical depth history.
        """

        QHII = self.history_asc['cgm_h_2']
        if 'igm_h_2' in self.history_asc:
            xHII = self.history_asc['igm_h_2']
        else:
            xHII = 0.0

        nH = self.cosm.nH(self.history_asc['z'])
        dldz = self.cosm.dldz(self.history_asc['z'])

        integrand = (QHII + (1. - QHII) * xHII) * nH

        if 'igm_he_1' in self.history_asc:
            QHeII = self.history_asc['cgm_h_2']
            xHeII = self.history_asc['igm_he_2']
            xHeIII = self.history_asc['igm_he_3']
            nHe = self.cosm.nHe(self.history_asc['z'])
            integrand += (QHeII + (1. - QHeII) * xHeII + 2. * xHeIII) \
                * nHe
        elif include_He:
            integrand *= (1. + self.cosm.y)

        integrand *= sigma_T * dldz

        tau = cumtrapz(integrand, self.history_asc['z'], initial=0)

        tau[self.history_asc['z'] > 100] = 0.0

        self.history_asc['tau_e'] = tau
        self.history['tau_e'] = tau[-1::-1]

        if self.history_asc['z'][0] < z_HeII_EoR:
            raise ValueError('Simulation ran past assumed HeII EoR! See z_HeII_EoR parameter.')

        # Make no arrays that go to z=0
        zlo, tlo = self.tau_post_EoR(include_He=include_He,
            z_HeII_EoR=z_HeII_EoR)

        tau_tot = tlo[-1] + tau

        tau_tot[self.history_asc['z'] > 100] = 0.0

        self.history_asc['z_CMB'] = np.concatenate((zlo, self.history_asc['z']))
        self.history['z_CMB'] = self.history_asc['z_CMB'][-1::-1]

        self.history_asc['tau_e_tot'] = np.concatenate((tlo, tau_tot))
        self.history['tau_e_tot'] = tau_tot[-1::-1]

    @property
    def tau_e(self):
        if not hasattr(self, '_tau_e'):
            if 'tau_e_tot' not in self.history:
                self.tau_CMB()

            z50 = np.argmin(np.abs(self.history['z_CMB'] - 50))
            self._tau_e = self.history['tau_e_tot'][z50]

        return self._tau_e

    def add_frequency_axis(self, ax):
        """
        Take plot with redshift on x-axis and add top axis with corresponding
        (observed) 21-cm frequency.

        Parameters
        ----------
        ax : matplotlib.axes.AxesSubplot instance
        """

        nu = np.arange(20, 220, 20)[-1::-1]
        nu_minor = np.arange(30, 230, 20)[-1::-1]
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

    def reverse_x_axis(self, ax, twinax=None):
        """
        By default, we plot quantities vs. redshift, ascending from left to
        right. This method will flip the x-axis, so that redshift is
        decreasing from left to right.
        """

        ax.invert_xaxis()

        if twinax is None:
            print("If you have a twinx axis, be sure to re-run add_time_axis or add_frequency_axis!")
            print("OR: pass that axis object as a keyword argument to this function!")
        else:
            twinax.invert_xaxis()

        pl.draw()

    def add_tau_inset(self, ax, inset=None, width=0.25, height=0.15, loc=4,
            mu=0.055, sig1=0.009, padding=0.02, borderpad=1, show_model=True,
            ticklabels=None, use_sigma_ticks=True, **kwargs):

        sig2 = 2. * sig1

        if inset is None:
            inset = self.add_inset(ax, inset=inset, width=width, height=height,
                loc=loc, mu=mu, sig1=sig1, padding=padding,
                borderpad=borderpad, **kwargs)

        inset.fill_between([mu-sig2-padding, mu-sig2], 0, 1,
            color='lightgray')
        inset.fill_between([mu+sig2+padding, mu+sig2], 0, 1,
            color='lightgray')
        if use_sigma_ticks:
            xticks = [mu-sig2-padding, mu-sig2, mu-sig1, mu,
                mu+sig1, mu+sig2, mu+sig2+padding]
            inset.set_xticks(xticks)
            inset.set_xticklabels(['', r'$-2\sigma$','',r'$\mu$', '', r'$+2\sigma$', ''])

        inset.set_title(r'$\tau_e$', fontsize=18, y=1.08)
        inset.xaxis.set_tick_params(width=1, length=5, labelsize=10)

        if show_model:
            inset.plot([self.tau_e]*2, [0, 1], **kwargs)

        pl.draw()

        return inset

    def add_inset(self, ax, inset=None, mu=None, sig1=None, lo=None, hi=None,
        width=0.3, height=0.15, loc=3,
        padding=0.02, borderpad=0.5, **kwargs):
        """
        Add inset 'slider' thing.
        """

        assert (mu is not None and sig1 is not None) \
            or (lo is not None or hi is not None)

        if inset is not None:
            pass
        else:
            inset = inset_locator.inset_axes(ax, width='{0}%'.format(100 * width),
                height='{0}%'.format(100 * height), loc=loc, borderpad=borderpad)

            inset.set_yticks([])
            inset.set_yticklabels([])

            if (lo is None and hi is None):
                sig2 = 2. * sig1
                lo = mu-sig2-padding
                hi = mu+sig2+padding

            inset.set_xlim(lo, hi)

        pl.draw()

        return inset

    def TemperatureHistory(self, **kwargs):
        return self.PlotTemperatureHistory(**kwargs)

    def PlotTemperatureHistory(self, ax=None, fig=1, show_Tcmb=False,
        show_Ts=False, show_Tk=True, scatter=False, mask=5,
        show_legend=False, **kwargs): # pragma: no cover
        """
        Plot kinetic, CMB, and spin temperatures vs. redshift.

        Parameters
        ----------

        """

        hasax = True
        if ax is None:
            hasax = False
            fig = pl.figure(fig)
            ax = fig.add_subplot(111)

        if 'label' not in kwargs:
            labels = [r'$T_{\gamma}$', r'$T_K$', r'$T_S$']
        else:
            labels = [kwargs['label']]*3
            del kwargs['label']

        if not scatter:
            if show_Tcmb:
                ax.semilogy(self.history['z'], self.cosm.TCMB(self.history['z']),
                    label=labels[0], **kwargs)

            if show_Tk:
                ax.semilogy(self.history['z'], self.history['igm_Tk'],
                    label=labels[1], **kwargs)

            if show_Ts:
                ax.semilogy(self.history['z'], self.history['igm_Ts'],
                    label=labels[2], **kwargs)
        else:
            if show_Tcmb:
                ax.scatter(self.history['z'][-1::-mask], self.cosm.TCMB(self.history['z'][-1::-mask]),
                    label=labels[0], **kwargs)

            ax.scatter(self.history['z'][-1::-mask], self.history['igm_Tk'][-1::-mask],
                label=labels[1], **kwargs)

            if show_Ts:
                ax.semilogy(self.history['z'][-1::-mask], self.history['igm_Ts'][-1::-mask],
                    label=labels[2], **kwargs)

        if not hasax:
            if hasattr(self, 'pf'):
                ax.set_xlim(int(self.pf["final_redshift"]),
                    self.pf["initial_redshift"])
            ax.set_xlabel(r'$z$')

        if not ax.get_ylabel():
            ax.set_ylabel(r'Temperature $(\mathrm{K})$')

        if show_legend:
            ax.legend(loc='best')

        pl.draw()

        return ax

    def IonizationHistory(**kwargs): # pragma: no cover
        return self.PlotIonizationHistory(**kwargs)

    def PlotIonizationHistory(self, ax=None, zone=None, element='h',
        fig=1, scatter=False, show_xhe_3=False,
        mask=5, show_xi=True, show_xe=True, show_xibar=False,
        show_legend=False, **kwargs): # pragma: no cover
        """
        Plot ionized fraction evolution.

        Parameters
        ----------
        zone : str
            Either "cgm", "igm", or None. If None, will plot mean ionized
            fraction evolution

        """

        hasax = True
        if ax is None:
            hasax = False
            fig = pl.figure(fig)
            ax = fig.add_subplot(111)

        if element == 'h':
            if zone is None:
                if self.pf['include_igm']:
                    xe = self.history['igm_{!s}_2'.format(element)]
                else:
                    xe = np.zeros_like(self.history['z'])
                if self.pf['include_cgm']:
                    xi = self.history['cgm_{!s}_2'.format(element)]
                else:
                    xi = np.zeros_like(self.history['z'])

                xavg = xi + (1. - xi) * xe

                to_plot = [xavg, xi, xe]
                show = [show_xibar, show_xi, show_xe]
            else:
                to_plot = [self.history['{0!s}_{1!s}_2'.format(zone, element)]]
                show = [True] * 2
                if show_xhe_3:
                    to_plot.append(self.history['{0!s}_{1!s}_3'.format(zone, element)])
                    show.append(True)
        else:
            to_plot = [self.history['igm_he_{}'.format(sp)] for sp in [2,3]]
            show = [True] * 2

        if 'label' not in kwargs:
            if element == 'h':
                labels = [r'$\bar{x}_i$', r'$x_i$', r'$x_e$']
            else:
                labels = [r'$x_{\mathrm{HeII}}$',
                          r'$x_{\mathrm{HeIII}}$']
        else:
            labels = [kwargs['label']]*len(to_plot)
            del kwargs['label']

        if 'ls' not in kwargs:
            ls = ['-', '--', ':']
        else:
            ls = [kwargs['ls']]*len(to_plot)
            del kwargs['ls']

        if not scatter:

            for i, element in enumerate(to_plot):
                if not show[i]:
                    continue

                ax.semilogy(self.history['z'], element, ls=ls[i], label=labels[i],
                    **kwargs)
        else:

            for i, element in enumerate(to_plot):
                if not show[i]:
                    continue

                ax.semilogy(self.history['z'][-1::-mask], element[-1::-mask],
                    label=labels[i], **kwargs)

        if hasattr(self, 'pf'):
            ax.set_xlim(int(self.pf["final_redshift"]),
                self.pf["initial_redshift"])
        ax.set_ylim(ax.get_ylim()[0], 1.5)

        ax.set_xlabel(r'$z$')
        ax.set_ylabel(r'Ionized Fraction')
        ax.set_yscale('log')

        if show_legend:
            ncol = 2 if element == 'he' else 1
            ax.legend(loc='best', ncol=ncol)

        pl.draw()

        return ax

    def IonizationRateHistory(self, **kwargs):
        return self.PlotIonizationRateHistory(**kwargs)

    def PlotIonizationRateHistory(self, fig=1, ax=None, species='h_1',
        show_legend=False, **kwargs): # pragma: no cover
        """
        Plot ionization rates as a function of redshift.
        """

        hasax = True
        if ax is None:
            hasax = False
            fig = pl.figure(fig)
            ax = fig.add_subplot(111)

        n1 = list(map(self.cosm.nH, self.history['z'])) if species == 'h_1' \
        else list(map(self.cosm.nHe, self.history['z']))

        n1 = np.array(n1) * np.array(self.history['igm_{!s}'.format(species)])

        ratePrimary = np.array(self.history['igm_Gamma_{!s}'.format(species)]) #* n1

        ax.semilogy(self.history['z'], ratePrimary, ls='-', **kwargs)

        rateSecondary = np.zeros_like(self.history['z'])

        for donor in ['h_1', 'he_1', 'he_2']:

            field = 'igm_gamma_{0!s}_{1!s}'.format(species, donor)

            if field not in self.history:
                continue

            n2 = list(map(self.cosm.nH, self.history['z'])) if donor == 'h_1' \
            else list(map(self.cosm.nHe, self.history['z']))

            n2 = np.array(n2) * np.array(self.history['igm_{!s}'.format(donor)])

            rateSecondary += \
                np.array(self.history['igm_gamma_{0!s}_{1!s}'.format(species, donor)]) * n2 / n1

        ax.semilogy(self.history['z'], rateSecondary, ls='--')

        ax.set_xlim(self.pf['final_redshift'], self.pf['initial_redshift'])

        pl.draw()

        return ax

    def OpticalDepthHistory(self, **kwargs):
        return self.PlotOpticalDepthHistory(**kwargs)

    def PlotOpticalDepthHistory(self, ax=None, fig=1,
        scatter=False, show_xi=True, show_xe=True, show_xibar=True,
        obs_mu=0.066, obs_sigma=0.012, show_obs=False, annotate_obs=False,
        include_He=True, z_HeII_EoR=3., **kwargs): # pragma: no cover
        """
        Plot (cumulative) optical depth to CMB evolution.

        Parameters
        ----------
        obs_mu : float
            Observationally constrained CMB optical depth. If supplied, will
            draw shaded region denoting this value +/- the error (obs_sigma,
            see below).
        obs_sigma : float
            1-sigma error on CMB optical depth.
        show_obs : bool
            Show shaded region for observational constraint?

        """

        hasax = True
        if ax is None:
            hasax = False
            fig = pl.figure(fig)
            ax = fig.add_subplot(111)

        if 'tau_e' not in self.history:
            self.tau_CMB(include_He=include_He, z_HeII_EoR=z_HeII_EoR)

        ax.plot(self.history_asc['z_CMB'], self.history_asc['tau_e_tot'], **kwargs)

        ax.set_xlim(0, 20)

        ax.set_xlabel(labels['z'])
        ax.set_ylabel(r'$\tau_e$')

        #ax.plot([0, self.history['z'].min()], [tau.max() + tau_post]*2, **kwargs)

        if show_obs:
            if obs_mu is not None:
                sig2 = get_nu(obs_sigma, 0.68, 0.95)
                ax.fill_between(ax.get_xlim(), [obs_mu-sig2]*2,
                    [obs_mu+sig2]*2, color='gray', alpha=0.2)
                ax.fill_between(ax.get_xlim(), [obs_mu-obs_sigma]*2,
                    [obs_mu+obs_sigma]*2, color='gray', alpha=0.5)

            if annotate_obs:
                #ax.annotate(r'$1-\sigma$ constraint',
                #    [self.history['z'].min(), obs_mu], ha='left',
                #    va='center')
                ax.annotate(r'plus post-EoR $\tau_e$',
                    [5, tau.max() + tau_post + 0.002], ha='left',
                    va='bottom')
                ax.set_title('daeshed lines are upper limits')

        ax.ticklabel_format(style='plain', axis='both')

        pl.draw()

        return ax

    def tau_post_EoR(self, include_He=True, z_HeII_EoR=3.):
        """
        Compute optical depth to electron scattering of the CMB.
        Equation 2 of Shull & Venkatesan 2008.

        Only includes optical depth to CMB contribution from a time after
        the final redshift in the dataset. It is assumed that the universe
        is fully ionized at all times, and that helium reionization occurs
        at z = 3.

        Returns
        -------
        Post-EoR (as far as our data is concerned) optical depth.
        """

        zmin = self.history['z'].min()
        ztmp = np.linspace(0, zmin, 1000)

        QHII = 1.0
        nH = self.cosm.nH(ztmp)
        dldz = self.cosm.dldz(ztmp)

        integrand = QHII * nH

        if 'igm_he_1' in self.history_asc:
            QHeII = 1.0
            xHeIII = 1.0
            nHe = self.cosm.nHe(ztmp)
            integrand += (QHeII + 2. * xHeIII) * nHe
        elif include_He:
            integrand *= (1. + self.cosm.y)

        integrand *= sigma_T * dldz

        tau = cumtrapz(integrand, ztmp, initial=0)

        return ztmp, tau

def add_redshift_axis(ax, twin_ax=None, zlim=80): # pragma: no cover
    """
    Take plot with frequency on x-axis and add top axis with corresponding
    redshift.

    Parameters
    ----------
    ax : matplotlib.axes.AxesSubplot instance
        Axis on which to add a redshift scale.
    """

    fig = ax.xaxis.get_figure()

    if zlim > 100:

        z = np.array([20, 40, 100, 400])[-1::-1]
        #z = np.arange(20, zlim, 40)[-1::-1]
        z_minor = np.arange(50, zlim, 50)[-1::-1]
        highz_labels = ['20', '30', '100']
    else:
        z = np.array([20, 40, 60, 80])[-1::-1]
        z_minor = np.arange(20, zlim, 10)[-1::-1]
        highz_labels = ['30', '80']

    #z_labels = list(map(str, z))
    lowz_labels = list(map(str, [6, 8, 10, 12, 15, 20]))

    z_labels = lowz_labels + highz_labels
    z = np.array(list(map(int, z_labels)))

    nu = nu_0_mhz / (1. + z)
    nu_minor = nu_0_mhz / (1. + z_minor)

    if twin_ax is None:
        ax_z = ax.twiny()
    else:
        ax_z = twin_ax

    ax_z.set_xlabel(labels['z'])
    ax_z.set_xticks(nu)
    ax_z.set_xticks(nu_minor, minor=True)

    # A bit hack-y
    #for i, label in enumerate(z_labels):
    #    if (zlim > 100) and (label not in highz_labels):
    #        z_labels[i] = ''
    #
    #    if (float(label) > 80) and (zlim < 100):
    #        z_labels[i] = ''

    ax_z.set_xticklabels(z_labels)
    ax_z.set_xlim(ax.get_xlim())

    pl.draw()

    return ax_z

def add_time_axis(ax, cosm, tlim=(100, 900), dt=200, dtm=50, tarr=None,
    tarr_m=None, rotation=0): # pragma: no cover
    """
    Take plot with redshift on x-axis and add top axis with corresponding
    time since Big Bang.

    Parameters
    ----------
    ax : matplotlib.axes.AxesSubplot instance
    """

    if tarr is None:
        t = np.arange(tlim[0], tlim[1]+dt, dt) # in Myr
        t_minor = np.arange(tlim[0]-dtm, tlim[1]+dtm, dtm)
    else:
        t = tarr
        t_minor = None

    _zt = np.array([cosm.z_of_t(tt * s_per_myr) for tt in t])
    _ztm = np.array([cosm.z_of_t(tt * s_per_myr) for tt in t_minor])

    ft = nu_0_mhz / (1. + _zt)
    ftm = nu_0_mhz / (1. + _ztm)
    zt = list(_zt)

    ax_time = ax.twiny()
   #if t_minor is not None:
   #    zt_minor = list(map(lambda tt: cosm.TimeToRedshiftConverter(0., tt * s_per_myr,
   #        np.inf), t_minor))
   #    ax_time.set_xticks(zt_minor, minor=True)

    ax_time.set_xlabel(r'$t \ \left[\mathrm{Myr} \right]$')

    # A bit hack-y
    time_labels = list(map(str, list(map(int, t))))
    for i, label in enumerate(time_labels):
        tnow = float(label)
        if (dt is None) and (dtm is None):
            if (tnow in [0,200,400,600,800]) or (tnow > 900):
                time_labels[i] = ''

    ax_time.set_xticks(ft)
    ax_time.set_xticks(ftm, minor=True)

    ax_time.set_xticklabels(time_labels, rotation=rotation)
    ax_time.set_xlim(ax.get_xlim())

    pl.draw()

    return ax_time
