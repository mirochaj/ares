"""

Global21cm.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sat Oct  3 14:57:31 PDT 2015

Description:

"""
import numpy as np
from ..util import labels
import matplotlib.pyplot as pl
from scipy.optimize import minimize
from ..physics.Constants import nu_0_mhz
from .TurningPoints import TurningPoints
from ..util.Math import central_difference
from matplotlib.ticker import ScalarFormatter
from ..analysis.BlobFactory import BlobFactory
from scipy.interpolate import interp1d, splrep, splev
from .MultiPhaseMedium import MultiPhaseMedium, add_redshift_axis, add_time_axis

class Global21cm(MultiPhaseMedium,BlobFactory):

    def __getattr__(self, name):
        """
        This gets called anytime we try to fetch an attribute that doesn't
        exist (yet).
        """

        # Trickery
        if hasattr(BlobFactory, name):
            return BlobFactory.__dict__[name].__get__(self, BlobFactory)

        if hasattr(MultiPhaseMedium, name):
            return MultiPhaseMedium.__dict__[name].__get__(self, MultiPhaseMedium)

        # Indicates that this attribute is being accessed from within a
        # property. Don't want to override that behavior!
        if (name[0] == '_'):
            raise AttributeError('This will get caught. Don\'t worry!')

        # Now, possibly make an attribute
        if name not in self.__dict__.keys():

            # See if this is a turning point
            spl = name.split('_')

            if len(spl) > 2:
                quantity = ''
                for item in spl[0:-1]:
                    quantity += '{!s}_'.format(item)
                quantity = quantity.rstrip('_')
                pt = spl[-1]
            else:
                try:
                    quantity, pt = spl
                except ValueError:
                    raise AttributeError('No attribute {!s}.'.format(name))

            if pt not in ['A', 'B', 'C', 'D', 'ZC', 'Bp', 'Cp', 'Dp']:
                # This'd be where e.g., zrei, should go
                raise NotImplementedError(('Looking for attribute ' +\
                    '\'{!s}\'.').format(name))

            if pt not in self.turning_points:
                return np.inf

            if quantity == 'z':
                self.__dict__[name] = self.turning_points[pt][0]
            elif quantity == 'nu':
                self.__dict__[name] = \
                    nu_0_mhz / (1. + self.turning_points[pt][0])
            elif quantity in self.history_asc:
                z = self.turning_points[pt][0]
                self.__dict__[name] = \
                    np.interp(z, self.history_asc['z'], self.history_asc[quantity])
            else:
                z = self.turning_points[pt][0]

                # Treat derivatives specially
                if quantity == 'slope':
                    self.__dict__[name] = self.derivative_of_z(z)
                elif quantity == 'curvature':
                    self.__dict__[name] = self.curvature_of_z(z)
                elif name in self.all_blob_names:
                    # Only works if scalar blob
                    self.__dict__[name] = self.get_blob(name)
                else:
                    raise KeyError('Unrecognized quantity: {!s}'.format(\
                        quantity))

        return self.__dict__[name]

    @property
    def dTbdz(self):
        if not hasattr(self, '_dTbdz'):
            self._z_p, self._dTbdz = \
                central_difference(self.history_asc['z'], self.history_asc['dTb'])
        return self._dTbdz

    @property
    def dTbdnu(self):
        if not hasattr(self, '_dTbdnu'):
            self._nu_p, self._dTbdnu = \
                central_difference(self.history['nu'], self.history['dTb'])
        return self._dTbdnu

    @property
    def dTb2dz2(self):
        if not hasattr(self, '_dTb2dz2'):
            self._z_pp, self._dTb2dz2 = \
                central_difference(self.z_p, self.dTbdz)
        return self._dTbdz

    @property
    def dTb2dnu2(self):
        if not hasattr(self, '_dTb2dnu2'):
            _dTbdnu = self.dTbdnu
            _nu = self._nu_p
            self._nu_pp, self._dTb2dnu2 = central_difference(_nu, _dTbdnu)
        return self._dTb2dnu2

    @property
    def nu_A(self):
        return nu_0_mhz / (1. + self.z_A)

    @property
    def z_A(self):
        if not hasattr(self, '_z_A'):

            if max(self.history_asc['z']) < 70:
                self._z_A = -np.inf
                return -np.inf

            zall = self.history_asc['z']
            Tall = self.history_asc['dTb']
            zfl = self.pf['first_light_redshift']
            zok = np.logical_and(zall > zfl, zall < 1e3)

            zguess = zall[zok][np.argmin(Tall[zok])]

            zslc = np.logical_and(zall[zok] > zguess - 2., zall[zok] < zguess + 2.)

            dz = 2.
            while zslc.sum() < 5:
                zslc = np.logical_and(zall[zok] > zguess - dz, zall[zok] < zguess + dz)
                dz += 1.

                if dz > 10:
                    break

            dTb = interp1d(zall[zok][zslc], Tall[zok][zslc], kind='cubic')

            try:
                to_min = minimize(lambda zz: dTb(zz), zguess)
                self._z_A = to_min.x[0]
            except:
                self._z_A = -np.inf

        return self._z_A

    @property
    def dTb_A(self):
        return np.interp(self.z_A, self.history_asc['z'],
            self.history_asc['dTb'])

    @property
    def z_p(self):
        if not hasattr(self, '_z_p'):
            tmp = self.dTbdz
        return self._z_p

    @property
    def z_pp(self):
        if not hasattr(self, '_z_p'):
            tmp = self.dTb2dz2
        return self._z_pp

    @property
    def nu_p(self):
        if not hasattr(self, '_nu_p'):
            tmp = self.dTbdnu
        return self._nu_p

    @property
    def nu_pp(self):
        if not hasattr(self, '_nu_pp'):
            tmp = self.dTb2dnu2
        return self._nu_pp

    @property
    def kurtosis(self):
        if not hasattr(self, '_kurtosis_abs'):
            ok = np.isfinite(self.history['dTb'])
            data = np.abs(self.history['dTb'][ok])
            self._kurtosis_abs = np.sum((data - np.mean(data))**4) \
                / float(data.size) / np.std(data)**4

        return self._kurtosis_abs

    @property
    def skewness(self):
        if not hasattr(self, '_skewness_abs'):
            ok = np.isfinite(self.history['dTb'])
            data = np.abs(self.history['dTb'][ok])
            self._skewness_abs = np.sum((data - np.mean(data))**3) \
                / float(data.size) / np.std(data)**3

        return self._skewness_abs

    @property
    def z_dec(self):
        if not hasattr(self, '_z_dec'):
            self._z_dec = self.cosm.z_dec
        return self._z_dec

    @property
    def Tk_dec(self):
        return np.interp(self.z_dec, self.history_asc['z'],
            self.history_asc['igm_Tk'])

    @property
    def track(self):
        if not hasattr(self, '_track'):
            if hasattr(self, 'pf'):
                self._track = TurningPoints(**self.pf)
            else:
                self._track = TurningPoints()
        return self._track

    def smooth_derivative(self, sm):
        arr = self.z_p[np.logical_and(self.z_p >= 6, self.z_p <= 45)]
        s = int(sm / np.diff(arr).mean())#self.pf['smooth_derivative']

        if s % 2 != 0:
            s += 1

        boxcar = np.zeros_like(self.dTbdz)
        boxcar[boxcar.size//2 - s//2: boxcar.size//2 + s//2] = \
            np.ones(s) / float(s)

        return np.convolve(self.dTbdnu, boxcar, mode='same')

    @property
    def turning_points(self):
        if not hasattr(self, '_turning_points'):

            _z = self.history['z']
            lowz = _z < 70

            # If we're here, the simulation has already been run.
            # We've got the option to smooth the derivative before
            # finding the extrema
            if self.pf['smooth_derivative'] > 0:
                _dTb = self.smooth_derivative(self.pf['smooth_derivative'])
            else:
                _dTb = self.history['dTb']

            z = self.history['z'][lowz]
            dTb = _dTb[lowz]

            # Otherwise, find them. Not the most efficient, but it gets the job done
            # Redshifts in descending order
            for i in range(len(z)):
                if i < 5:
                    continue

                stop = self.track.is_stopping_point(z[0:i], dTb[0:i])

            # See if anything is wonky
            fixes = {}
            if 'C' in self.track.turning_points:
                zC = self.track.turning_points['C'][0]
                if (zC < 0) or (zC > 50):
                    print("WARNING: absorption minimum redshift wonky (z={})".format(zC))
                    i_min = np.argmin(self.history['dTb'])
                    fixes['C'] = (self.history['z'][i_min],
                        self.history['dTb'][i_min], -99999)
                    print("WARNING: Reset to z={}, dTb={}".format(*fixes['C'][0:2]))

            if 'D' in self.track.turning_points:
                zD = self.track.turning_points['D'][0]
                TD = self.track.turning_points['D'][1]
                if (zD < 0) or (zD > 50):
                    i_max = np.argmax(self.history['dTb'])
                    fixes['D'] = (self.history['z'][i_max],
                        self.history['dTb'][i_max], -99999)
                elif TD < 1e-4:
                    fixes['D'] = (-np.inf, -np.inf, -99999)

            result = self.track.turning_points
            result.update(fixes)

            self._turning_points = result

            ##
            # If there are multiple extrema (e.g, C and C'), fix order.
            ##

            # Don't do this just yet. Going to break compatibility with
            # some recent datasets.

            #for tp in list('BCD'):
            #    if '{!s}p'.format(tp) in self.turning_points:
            #        tmp_p = self.turning_points['{!s}p'.format(tp)]
            #        tmp = self.turning_points[tp]
            #
            #        del self.turning_points['{!s}p'.format(tp)]
            #        del self.turning_points[tp]
            #
            #        self.turning_points[tp] = tmp_p
            #        self.turning_points['{!s}p'.format(tp)] = tmp

        return self._turning_points

    def derivative_of_freq(self, freq):
        interp = interp1d(self.nu_p, self.dTbdnu, kind='linear',
            bounds_error=False, fill_value=-np.inf)
        return interp(freq)

    def curvature_of_freq(self, freq):
        interp = interp1d(self.nu_pp, self.dTb2dnu2, kind='linear',
            bounds_error=False, fill_value=-np.inf)
        return interp(freq)

    def derivative_of_z(self, z):
        freq = nu_0_mhz / (1. + z)
        return self.derivative_of_freq(freq)

    def curvature_of_z(self, z):
        freq = nu_0_mhz / (1. + z)
        return self.curvature_of_freq(freq)

    def PlotSaturatedLimit(self, ax): # pragma: no cover
        z = nu_0_mhz / self.history['nu'] - 1.
        dTb = self.hydr.saturated_limit(z)

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        ax.plot(self.history['nu'], dTb, color='k', ls=':')
        ax.fill_between(self.history['nu'], dTb, 500 * np.ones_like(dTb),
            color='none', hatch='X', edgecolor='k', linewidth=0.0)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        pl.draw()

        return ax

    def PlotAdiabaticFloor(self, ax, gap=None, temp_units='mk',
        **kwargs): # pragma: no cover
        z = nu_0_mhz / self.history['nu'] - 1.
        dTb = self.hydr.adiabatic_floor(z)

        if temp_units.lower() in ['k', 'kelvin']:
            conv = 1e-3
        else:
            conv = 1.

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        nu = self.history['nu']
        dTb = dTb

        if kwargs == {}:
            ax.fill_between(nu,
                -2e3 * np.ones_like(dTb) * conv, dTb * conv,
                color='none', hatch='X', edgecolor='k', linewidth=0.)
        else:
            if gap is None:
                ax.plot(nu, dTb * conv, **kwargs)
            else:
                i1 = np.argmin(np.abs(nu - gap[0]))
                i2 = np.argmin(np.abs(nu - gap[1]))

                ax.plot(nu[0:i1], dTb[0:i1] * conv, **kwargs)
                ax.plot(nu[i2:], dTb[i2:] * conv, **kwargs)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        pl.draw()

        return ax

    def GlobalSignature(self, **kwargs): # pragma: no cover
        """
        Wrapper around `Plot21cmGlobalSignal` for backward compatibility.
        """
        return self.Plot21cmGlobalSignal(**kwargs)

    def Plot21cmGlobalSignal(self, ax=None, fig=1, freq_ax=False,
        time_ax=False, z_ax=True, mask=None, scatter=False, xaxis='nu',
        ymin=None, ymax=50, zmax=None, rotate_xticks=False, rotate_yticks=False,
        force_draw=False, zlim=80, temp_unit='mK', yscale='linear',
        take_abs=False, **kwargs): # pragma: no cover
        """
        Plot differential brightness temperature vs. redshift (nicely).

        Parameters
        ----------
        ax : matplotlib.axes.AxesSubplot instance
            Axis on which to plot signal.
        fig : int
            Figure number.
        freq_ax : bool
            Add top axis denoting corresponding (observed) 21-cm frequency?
        time_ax : bool
            Add top axis denoting corresponding time since Big Bang?
        z_ax : bool
            Add top axis denoting corresponding redshift? Only applicable
            if xaxis='nu' (see below).
        scatter : bool
            Plot signal as scatter-plot?
        mask : int
            If scatter==True, this defines the sampling "rate" of the data,
            i.e., only every mask'th element is included in the plot.
        xaxis : str
            Determines whether x-axis is redshift or frequency.
            Options: 'z' or 'nu'

        Returns
        -------
        matplotlib.axes.AxesSubplot instance.

        """

        if xaxis == 'nu' and freq_ax:
            freq_ax = False
        if xaxis == 'z' and z_ax:
            z_ax = False

        if ax is None:
            gotax = False
            fig = pl.figure(fig)
            ax = fig.add_subplot(111)
        else:
            gotax = True

        conv = 1.
        if temp_unit.lower() in ['k', 'kelvin']:
            conv = 1e-3

        if mask is not None:
            nu_plot, dTb_plot = \
                self.history[xaxis][mask], self.history['dTb'][mask] * conv
        else:
            nu_plot, dTb_plot = \
                self.history[xaxis], self.history['dTb'] * conv

        if take_abs:
            dTb_plot = np.abs(dTb_plot)

        ##
        # Plot the stupid thing
        ##
        if scatter is False:
            ax.plot(nu_plot, dTb_plot, **kwargs)
        else:
            ax.scatter(self.history[xaxis][-1::-mask],
                self.history['dTb'][-1::-mask] * conv, **kwargs)

        if zmax is None:
            zmax = self.pf["initial_redshift"]

        zmin = self.pf["final_redshift"] if self.pf["final_redshift"] >= 10 \
            else 5

        # x-ticks
        if xaxis == 'z' and hasattr(self, 'pf'):
            xticks = list(np.arange(zmin, zmax, zmin))
            xticks_minor = list(np.arange(zmin, zmax, 1))
        else:
            xticks = np.arange(0, 250, 50)
            xticks_minor = np.arange(10, 200, 10)

        # Some elements deemed objects when run through pipelines...
        dTb = np.array(self.history['dTb'], dtype=float)

        if ymin is None and yscale == 'linear':
            ymin = max(min(min(dTb[np.isfinite(dTb)]), ax.get_ylim()[0]), -500)

            # Set lower y-limit by increments of 50 mK
            for val in [-50, -100, -150, -200, -250, -300, -350, -400, -450, -500, -550, -600]:
                if val <= ymin:
                    ymin = int(val)
                    break

        if ymax is None:
            ymax = max(max(dTb[np.isfinite(dTb)]), ax.get_ylim()[1])

        if yscale == 'linear':
            if (not gotax) or force_draw:
                yticks = np.arange(int(ymin / 50) * 50, 100, 50) * conv
                ax.set_yticks(yticks)
            else:
                # Minor y-ticks - 10 mK increments
                yticks = np.linspace(ymin, 50, int((50 - ymin) / 10. + 1)) * conv
                yticks = list(yticks)

                # Remove major ticks from minor tick list
                if ymin >= -200:
                    for y in np.linspace(ymin, 50, int((50 - ymin) / 50. + 1)) * conv:
                        if y in yticks:
                            yticks.remove(y)

                ax.set_ylim(ymin * conv, ymax * conv)
                ax.set_yticks(yticks, minor=True)

        if xaxis == 'z' and hasattr(self, 'pf'):
            ax.set_xlim(5, self.pf["initial_redshift"])
        else:
            ax.set_xlim(0, 210)

        if (not gotax) or force_draw:
            ax.set_xticks(xticks, minor=False)
            ax.set_xticks(xticks_minor, minor=True)


            xt = []
            for x in ax.get_xticklabels():
                xt.append(x.get_text())

            ax.set_xticklabels(xt, rotation=45. if rotate_xticks else 0)

            yt = []
            for y in ax.get_yticklabels():
                if not y.get_text().strip():
                    break
                yt.append(y.get_text())

            if yt == []:
                yt = yticks

            ax.set_yticklabels(yt, rotation=45. if rotate_yticks else 0)

        if ax.get_xlabel() == '':
            if xaxis == 'z':
                ax.set_xlabel(labels['z'], fontsize='x-large')
            else:
                ax.set_xlabel(labels['nu'])

        if ax.get_ylabel() == '':
            if temp_unit.lower() == 'mk':
                ax.set_ylabel(labels['dTb'], fontsize='x-large')
            else:
                ax.set_ylabel(r'$\delta T_b \ (\mathrm{K})$', fontsize='x-large')

        # Twin axes along the top
        if freq_ax:
            twinax = self.add_frequency_axis(ax)
        elif time_ax:
            twinax = add_time_axis(ax, self.cosm)
        elif z_ax:
            twinax = add_redshift_axis(ax, zlim=zmax)
        else:
            twinax = None

        self.twinax = twinax

        if gotax and (ax.get_xlabel().strip()) and (not force_draw):
            pl.draw()
            return ax, twinax

        try:
            ax.ticklabel_format(style='plain', axis='both')
        except AttributeError:
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_major_formatter(ScalarFormatter())
            #twinax.xaxis.set_major_formatter(ScalarFormatter())
            ax.ticklabel_format(style='plain', axis='both')

        pl.draw()

        return ax, twinax

    def GlobalSignatureDerivative(self, **kwargs): # pragma: no cover
        """
        Wrapper around `Plot21cmGlobalSignalDerivative` for backward
        compatibility.
        """
        return self.Plot21cmGlobalSignalDerivative(**kwargs)

    def Plot21cmGlobalSignalDerivative(self, ax=None, fig=1,
        show_signal=False, **kwargs): # pragma: no cover
        """
        Plot signal and its first derivative (nicely).

        Parameters
        ----------

        Returns
        -------

        """

        if show_signal:
            if ax is None:
                gotax = False
                fig, axes = pl.subplots(2, 1, num=fig)
                ax = axes[1]
            else:
                gotax = True
        else:
            if ax is None:
                gotax = False
                fig = pl.figure(fig)
                ax = fig.add_subplot(111)
                axes = [ax]
            else:
                gotax = True

        if show_signal:
            ax2 = self.GlobalSignature(ax=axes[0], z_ax=False, **kwargs)
            axes[1].plot(self.nu_p, self.dTbdnu, **kwargs)
            axes[1].set_xticks(axes[0].get_xticks())
            axes[1].set_xticklabels([])
            axes[1].set_xlim(axes[0].get_xlim())
        else:
            ax.plot(self.nu_p, self.dTbdnu, **kwargs)

        if not gotax:
            if not show_signal:
                ax.set_xlabel(labels['nu'])

            ax.set_ylabel(r'$\delta T_{\mathrm{b}}^{\prime} \ (\mathrm{mK} \ \mathrm{MHz}^{-1})$')

        pl.draw()

        if show_signal:
            return axes
        else:
            return ax

    def add_Ts_inset(self, ax, inset=None, width=0.3, height=0.15, loc=3,
            z=8.4, lo=1.9, hi=None, padding=0.02, borderpad=0.5,
            **kwargs): # pragma: no cover

        if inset is None:
            inset = self.add_inset(ax, inset=inset, width=width, height=height,
                loc=loc, lo=lo, hi=hi, padding=padding, mu=None, sigma=None,
                borderpad=borderpad, **kwargs)

        inset.fill_between([0., self.cosm.Tgas(z)], 0, 1,
            color='k', facecolor='none', hatch='//')
        #inset.plot([self.cosm.Tgas(z)]*2, [0, 1],
        #    color='k')
        inset.fill_between([self.cosm.Tgas(z), lo], 0, 1,
            color='lightgray')
        #inset.fill_between([self.cosm.TCMB(z), hi], 0, 1,
        #    color='k', facecolor='none', hatch='-')
        xticks = [1, 10, 100]
        inset.set_xticks(xticks)

        if hi is None:
            hi = max(self.history['Ts'])

        inset.set_xlim(0.8, hi)

        inset.set_title(r'$T_S(z={0:.2g})$'.format(z), fontsize=16, y=1.08)
        inset.xaxis.set_tick_params(width=1, length=5, labelsize=10)

        Ts = np.interp(z, self.history_asc['z'], self.history_asc['igm_Ts'])
        inset.plot([Ts]*2, [0, 1], **kwargs)
        inset.set_xscale('log')

        pl.draw()

        return inset

    def fill_between_sims(self, other_sim, ax=None, **kwargs):
        sims = [self, other_sim]

        assert len(sims) == 2, 'Only works for sets of two simulations.'


        nu = []; dTb = []; C = []; D = []

        for i, sim in enumerate(sims):

            nu.append(sim.history['nu'])
            dTb.append(sim.history['dTb'])

            #ax = sim.GlobalSignature(ax=ax, **kwargs)

            #C.append(sim.turning_points['C'])
            #D.append(sim.turning_points['D'])

        y1_w_x0 = np.interp(nu[0], nu[1], dTb[1])
        ax.fill_between(nu[0], dTb[0], y1_w_x0, **kwargs)

        #for tp in [C, D]:
        #    nu_C_0 = nu_0_mhz / (1. + tp[0][0])
        #    nu_C_1 = nu_0_mhz / (1. + tp[1][0])
        #    T_C_0 = tp[0][1]
        #    T_C_1 = tp[1][1]
        #
        #    nu_min = min(nu_C_0, nu_C_1)
        #
        #    # Line connecting turning points
        #    def y_min(nu):
        #
        #        dnu = abs(nu_C_0 - nu_C_1)
        #        dT = abs(T_C_0 - T_C_1)
        #        m = dT / dnu
        #
        #        return m * (nu - nu_min) + min(T_C_0, T_C_1)
        #
        #    new_nu = np.linspace(min(nu_C_0, nu_C_1), max(nu_C_0, nu_C_1))
        #
        #    new_T0 = np.interp(new_nu, nu[0], dTb[0])
        #    new_T1 = np.interp(new_nu, nu[1], dTb[1])
        #
        #    if tp == C:
        #        ax.fill_between(new_nu, y_min(new_nu), np.minimum(new_T0, new_T1),
        #            **kwargs)
        #    else:
        #        ax.fill_between(new_nu, y_min(new_nu), np.maximum(new_T0, new_T1),
        #            **kwargs)

        pl.draw()

        return ax

    def Slope(self, freq):
        """
        Return slope of signal in mK / MHz at input frequency (MHz).
        """

        return np.interp(freq, self.nu_p, self.dTbdnu)

    def WidthMeasure(self, max_fraction=0.5, peak_relative=False, to_freq=True,
        absorption=True):
        # This only helps with backward compatibility between two obscure
        # revisions that probably nobody is using...
        return self.Width(max_fraction, peak_relative, to_freq)

    def Width(self, max_fraction=0.5, peak_relative=False, to_freq=True,
        absorption=True, dark_ages=False):
        """
        Return a measurement of the width of the absorption or emission signal.

        Parameters
        ----------
        max_fraction : float
            At what fraction of the peak should we evaluate the width?
        peak_relative: bool
            If True, compute the width on the left (L) and right (R) side of
            the peak separately, and return R - L. If Not, the return value is
            the full width of the peak evaluated at max_fraction * its max.
        to_freq: bool
            If True, return value is in MHz. If False, it is a differential
            redshift element.
        absorption : bool
            If True, assume absorption signal, otherwise, use emission signal.

        .. note :: With default parameters, this function returns the
            full-width at half-maximum (FWHM) of the absorption signal.

        """

        if absorption:
            if dark_ages:
                tp = 'A'
            else:
                tp = 'C'
        else:
            tp = 'D'

        if tp == 'A':
            z_pt = self.z_A
            n_pt = self.nu_A
            T_pt = self.dTb_A
        else:


        #if tp not in self.turning_points:
        #    return -np.inf

            z_pt = self.turning_points[tp][0]
            n_pt = nu_0_mhz / (1. + z_pt)
            T_pt = self.turning_points[tp][1]

        if not np.isfinite(z_pt):
            return -np.inf

        # Only use low redshifts once source are "on"
        _z = self.history_asc['z']
        ok = _z < self.pf['initial_redshift']
        z = self.history_asc['z'][ok]
        dTb = self.history_asc['dTb'][ok]

        # (closest) index corresponding to the extremum of interest.
        # Using amplitude can lead to errors when heating trough is comparable
        # in amplitude to dark ages trough
        i_max = np.argmin(np.abs(z - z_pt))

        # At what fraction of peak do we measure width?
        f_max = max_fraction * T_pt

        if len(dTb[:i_max]) < 2 or len(dTb[i_max:]) < 2:
            return -np.inf

        # Need to restrict range to avoid double-valued-ness...? Might as well.
        if dark_ages:
            i_hi = np.argmin(np.abs(z - max(z)))

            print(i_max, i_hi, z[i_max], z[i_hi])
        elif absorption:
            if 'B' in self.turning_points:
                if np.isfinite(self.turning_points['B'][0]):
                    i_hi = np.argmin(np.abs(z - self.turning_points['B'][0]))
                else:
                    i_hi = np.argmin(np.abs(z - max(z)))
            else:
                i_hi = np.argmin(np.abs(z - max(z)))

        else:
            i_hi = None

        # Don't fully understand this
        if (i_hi is not None) and (i_hi < i_max):
            return -np.inf

        # I think this only happens when absorption signal is basically
        # negligible.
        if len(dTb[:i_max]) < 2:
            return -np.inf
        if len(dTb[i_max:i_hi]) < 2:
            return -np.inf

        # Break the data into two intervals: redshifts above and below
        # the extremum. Interpolate to find desired point.
        # I've experimented with cubic/quadratic and most of the time they
        # work fine but in some cases they give nonsensical results for no
        # apparent reason, hence the linear interpolation.
        interp_l = interp1d(dTb[:i_max], z[:i_max],
            bounds_error=False, fill_value=-np.inf, kind='linear')
        # At this point, the signal is getting more negative
        interp_r = interp1d(dTb[i_max:i_hi], z[i_max:i_hi],
            bounds_error=False, fill_value=-np.inf, kind='linear')

        # The "l" and "r" are in terms of redshift.

        # Interpolate to find redshifts where f_max occurs
        l = abs(interp_l(f_max))
        r = abs(interp_r(f_max))

        if np.any(np.isinf([l, r])):
            return -np.inf

        # "l" and "r" are now backwards since we're going to frequency
        if to_freq:
            l = nu_0_mhz / (1. + l)
            r = nu_0_mhz / (1. + r)

        if peak_relative:
            if to_freq:
                l = abs(n_pt - l)
                r = abs(n_pt - r)
            else:
                l = abs(z_pt[i] - l)
                r = abs(z_pt[i] - r)

            val = -(r - l)
        else:
            val = abs(r - l)

        return val
