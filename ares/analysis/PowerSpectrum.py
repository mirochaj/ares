import numpy as np
from ..util import labels
import matplotlib.pyplot as pl
from .Global21cm import Global21cm
from ..util.Misc import split_by_sign
from scipy.interpolate import interp1d
from ..physics.Constants import nu_0_mhz
from matplotlib.ticker import ScalarFormatter
from ..analysis.BlobFactory import BlobFactory
from .MultiPhaseMedium import MultiPhaseMedium

# Distinguish between mean history and fluctuations?
class PowerSpectrum(MultiPhaseMedium,BlobFactory): # pragma: no cover
    def __init__(self, data=None, suffix='fluctuations', **kwargs):
        """
        Initialize analysis object.

        Parameters
        ----------
        data : dict, str
            Either a dictionary containing the entire history or the prefix
            of the files containing the history/parameters.

        """

        MultiPhaseMedium.__init__(self, data=data, suffix=suffix, **kwargs)

    @property
    def redshifts(self):
        if not hasattr(self, '_redshifts'):
            self._redshifts = self.history['z']
        return self._redshifts

    @property
    def gs(self):
        if not hasattr(self, '_gs'):
            if hasattr(self, 'prefix'):
                self._gs = Global21cm(data=self.prefix)
            elif 'dTb' in self.history:
                hist = {'z': self.history['z'], 'dTb': self.history['dTb']}
                self._gs = Global21cm(data=hist)
            else:
                raise AttributeError('Cannot initialize Global21cm instance!')

        return self._gs

    def PowerSpectrum(self, z, field='21', ax=None, fig=1,
        force_draw=False, dimensionless=True, take_sqrt=False,
        scatter=False, renorm=None, **kwargs):
        """
        Plot differential brightness temperature vs. redshift (nicely).

        Parameters
        ----------
        ax : matplotlib.axes.AxesSubplot instance
            Axis on which to plot signal.
        fig : int
            Figure number.

        Returns
        -------
        matplotlib.axes.AxesSubplot instance.

        """

        if ax is None:
            gotax = False
            fig = pl.figure(fig)
            ax = fig.add_subplot(111)
        else:
            gotax = True

        iz = np.argmin(np.abs(z - self.redshifts))

        k = self.history['k']

        ps_s = 'ps_%s' % field
        if dimensionless and ('%s_dl' % ps_s in self.history):
            ps = self.history['%s_dl' % ps_s][iz]
            if take_sqrt:
                ps = np.sqrt(ps)

        elif dimensionless:
            if field == '21':
                if renorm is None:
                    dTb0 = self.history['dTb0_1'][iz]
                else:
                    dTb0 = renorm[iz]

                norm = dTb0**2
            else:
                print("dunno norm for field={}".format(field))
                norm = 1.

            ps = norm * self.history[ps_s][iz] * k**3 / 2. / np.pi**2

            if take_sqrt:
                ps = np.sqrt(ps)
        else:
            ps = self.history[ps_s][iz]

        if scatter:
            ax.scatter(k, ps, **kwargs)
        else:
            ax.loglog(k, ps, **kwargs)

        if gotax and (ax.get_xlabel().strip()) and (not force_draw):
            return ax

        if ax.get_xlabel() == '':
            ax.set_xlabel(labels['k'], fontsize='x-large')

        if ax.get_ylabel() == '':
            if dimensionless and ('%s_dl' % field in self.history):
                ps = self.history['%s_dl' % field]
            elif dimensionless:
                if take_sqrt:
                    ax.set_ylabel(r'$\Delta_{21}(k)$', fontsize='x-large')
                elif field == '21':
                    ax.set_ylabel(labels['dpow'], fontsize='x-large')
                else:
                    ax.set_ylabel(r'$\Delta^2(k)$', fontsize='x-large')
            else:
                ax.set_ylabel(labels['pow'], fontsize='x-large')

        ax.set_xlim(1e-2, 1e1)
        ax.set_ylim(1e-3, 1e4)

        pl.draw()

        return ax

    def CorrelationFunction(self, z, field='ii', ax=None, fig=1,
        force_draw=False, **kwargs):
        """
        Plot correlation function of input fields.

        Parameters
        ----------
        ax : matplotlib.axes.AxesSubplot instance
            Axis on which to plot signal.
        fig : int
            Figure number.

        Returns
        -------
        matplotlib.axes.AxesSubplot instance.

        """

        if ax is None:
            gotax = False
            fig = pl.figure(fig)
            ax = fig.add_subplot(111)
        else:
            gotax = True

        iz = np.argmin(np.abs(z - self.redshifts))

        cf_s = 'cf_%s' % field

        if cf_s not in self.history:
            # Short-cuts
            if field == 'ii':
                if 'cf_ii' in self.history:
                    cf = self.history['cf_ii'][iz]
                else:
                    cf = -self.history['cf_xx'][iz]
            elif field == 'xx':
                if 'cf_xx' in self.history:
                    cf = self.history['cf_xx'][iz]
                else:
                    cf = self.history['cf_ii'][iz]
            elif field == 'xd':
                cf = -self.history['cf_id'][iz]
            elif field == 'mm':
                cf = self.history['cf_dd'][iz]
            else:
                raise KeyError('Field `{}` not found in history.'.format(field))
        else:
            cf = self.history[cf_s][iz]

        R = self.history['R']

        x_ch, cf_chunks = split_by_sign(R, cf)

        ct = 0
        for j, chunk in enumerate(cf_chunks):
            if np.all(chunk < 0):
                lw = 1
            else:
                lw = 3

            tmp = kwargs.copy()
            tmp['lw'] = lw

            if ct == 0 and 'label' in kwargs:
                tmp['label'] = kwargs['label']
            else:
                tmp['label'] = None

            ax.loglog(x_ch[j], np.abs(chunk), **tmp)
            ct += 1

        if gotax and (ax.get_xlabel().strip()) and (not force_draw):
            return ax

        if ax.get_xlabel() == '':
            ax.set_xlabel(r'$r \ [\mathrm{cMpc}]$', fontsize='x-large')

        if ax.get_ylabel() == '':
            s = r'$\xi_{%s}$' % field
            ax.set_ylabel(s, fontsize='x-large')

        if 'label' in kwargs:
            if kwargs['label'] is not None:
                ax.legend(loc='best')

        pl.draw()

        return ax

    def BubbleSizeDistribution(self, z, ax=None, fig=1, force_draw=False,
        by_volume=False, region='i', **kwargs):
        """
        Plot bubble size distribution.

        Parameters
        ----------
        ax : matplotlib.axes.AxesSubplot instance
            Axis on which to plot signal.
        fig : int
            Figure number.
        by_volume : bool
            If True, uses bubble volume rather than radius.

        Returns
        -------
        matplotlib.axes.AxesSubplot instance.

        """

        if ax is None:
            gotax = False
            fig = pl.figure(fig)
            ax = fig.add_subplot(111)
        else:
            gotax = True

        iz = np.argmin(np.abs(z - self.redshifts))

        if region == 'i':

            if 'r_i' not in self.history:
                return

            R = self.history['r_i'][iz]
            M = self.history['m_i'][iz]
            bsd = self.history['n_i'][iz]
            delta_B = self.history['delta_B'][iz]
            Q = self.history['Qi'][iz]
        else:

            if 'r_h' not in self.history:
                return

            R = self.history['r_i'][iz]
            M = self.history['m_i'][iz]
            bsd = self.history['n_i'][iz]
            delta_B = self.history['delta_B_h'][iz]
            Q = self.history['Qh'][iz]

        rho0 = self.cosm.mean_density0 * self.cosm.fbaryon

        dvdr = 4. * np.pi * R**2
        dmdr = rho0 * (1. + delta_B) * dvdr
        dmdlnr = dmdr * R
        dndlnR = bsd * dmdlnr

        V = 4. * np.pi * R**3 / 3.

        if by_volume:
            dndV = bsd * dmdr / dvdr
            ax.loglog(V, V * dndV, **kwargs)
            ax.set_xlabel(r'$V \ [\mathrm{Mpc}^{3}]$')
            ax.set_ylabel(r'$V \ dn/dV$')
            ax.set_xlim(1, 1e6)
            ax.set_ylim(1e-8, 1e-2)
        else:
            ax.semilogx(R, V * dndlnR / Q, **kwargs)
            ax.set_xlabel(r'$R \ [\mathrm{Mpc}]$')
            ax.set_ylabel(r'$Q^{-1} V \ dn/dlnR$')
            ax.set_yscale('linear')
            ax.set_ylim(0, 0.7) # Split difference between FZH04 and PC14

        pl.draw()

        return ax

    def BubbleFillingFactor(self, ax=None, fig=1, force_draw=False,
        **kwargs):
        """
        Plot the fraction of the volume composed of ionized bubbles.
        """
        if ax is None:
            gotax = False
            fig = pl.figure(fig)
            ax = fig.add_subplot(111)
        else:
            gotax = True

        Qall = []
        for i, z in enumerate(self.redshifts):
            Qall.append(self.history['Qi'][i])

        ax.plot(self.redshifts, Qall, **kwargs)
        ax.set_xlabel(r'$z$')
        ax.set_ylabel(r'$Q_{\mathrm{HII}}$')
        ax.set_ylim(0, 1)

        pl.draw()

        return ax

    def RedshiftEvolution(self, field='21', k=0.2, ax=None, fig=1,
        dimensionless=True, show_gs=False, mp_kwargs={}, scatter=True,
        scatter_kwargs={}, renorm=None, orientation='vertical', show_dd=False,
        **kwargs):
        """
        Plot the fraction of the volume composed of ionized bubbles.

        Parameters
        ----------

        """

        if ax is None:
            gotax = False
            if show_gs:
                if mp_kwargs == {}:
                    mp_kwargs = {'fig': fig}

                if orientation == 'vertical':
                    dims = (2, 1)
                    mp_kwargs['padding'] = (0, 0.1)
                else:
                    dims = (1, 2)
                    mp_kwargs['padding'] = (0.3, 0.)

                fig, axes = pl.subplots(2, 1, num=fig)
            else:
                fig = pl.figure(fig)
                ax = fig.add_subplot(111)
                axes = [ax]
        else:
            gotax = True

        p = []
        dd = []
        for i, z in enumerate(self.redshifts):
            if dimensionless and 'ps_21_dl' in self.history:
                pow_z = self.history['ps_21_dl'][i]
            else:
                pow_z = self.history['ps_%s' % field][i]

            p.append(np.interp(np.log(k), np.log(self.history['k']), pow_z))

            if show_dd:
                dd.append(np.interp(np.log(k), np.log(self.history['k']),
                    self.history['ps_mm'][i]))

        p = np.array(p)

        if dimensionless and ('ps_21_dl' in self.history):
            ps = p
        elif dimensionless:
            if renorm is None:
                dTb0 = self.history['dTb0_1']
            else:
                dTb0 = renorm

            norm = dTb0**2

            ps = norm * p * k**3 / 2. / np.pi**2

            if show_dd:
                ps_dd = norm * np.array(dd) * k**3 / 2. / np.pi**2
                ps_dd *= (1. - self.history['Qi'])**2
        else:
            ps = p

        if show_gs:
            ax1 = axes[0]
        else:
            ax1 = ax

        if scatter:
            ax1.scatter(self.redshifts, ps, **scatter_kwargs)
        else:
            ax1.plot(self.redshifts, ps, **kwargs)

        if show_dd:
            if 'label' in kwargs:
                del kwargs['label']
            ax1.plot(self.redshifts, ps_dd, lw=1, **kwargs)

        ax1.set_xlim(min(self.redshifts), max(self.redshifts))
        ax1.set_yscale('log')
        ax1.set_xlim(6, 20)
        ax1.set_ylim(1e-4, 1e4)

        if (not gotax):
            ax1.set_xlabel(r'$z$')

        if dimensionless:
            ax1.set_ylabel(labels['dpow'])
        else:
            ax1.set_ylabel(labels['pow'])

        if show_gs:
            self.gs.GlobalSignature(ax=mp.grid[1], xaxis='z', **kwargs)
            axes[1].set_xlim(6, 20)

            if orientation == 'vertical' and (not gotax):
                axes[1].set_xticklabels([])
                axes[1].set_xlabel('')

        pl.draw()

        if show_gs:
            return axes
        else:
            return ax1

    def _split_cf(self, z, key):
        """
        Split apart a correlation function into its positive and negative
        chunks.
        """

        iz = np.argmin(np.abs(z - self.redshifts))

        if type(key) is str:
            data = self.history[key][iz]
        else:
            data = key

        if 'cf' in key:
            x = self.history['R']
        else:
            x = self.history['k']

        return split_by_sign(x, data)

        # Might be multiple sign changes

        #splitter = np.diff(np.sign(data))
        #
        #dr = self.history['R']
        #
        #if np.all(splitter == 0):
        #    chunks = [data]
        #    dr_ch = [dr]
        #else:
        #    splits = np.atleast_1d(np.argwhere(splitter != 0).squeeze()) + 1
        #    chunks = np.split(data, splits)
        #    dr_ch = np.split(dr, splits)
        #
        #return dr_ch, chunks

    #def CheckFluctuations(self, redshifts, include_xcorr=False, real_space=True,
    #    split_by_scale=False, include_fields=['mm','ii','coco','21_s','21'],
    #    colors=['k','b','g','c','m','r'], mp_kwargs={}, mp=None, dimensionless=True,
    #    **kwargs):
    #    """
    #    Plot various constituent correlation functions (or power spectra).
    #    """
#
    #    if mp is None:
    #        mp = MultiPanel(dims=(1+include_xcorr, len(redshifts)),
    #            padding=(0.25, 0.15), fig=fig, **mp_kwargs)
    #        gotmp = False
    #    else:
    #        gotmp = True
#
    #    if real_space:
    #        prefix = 'cf'
    #        x = self.history['R']
    #    else:
    #        prefix = 'ps'
    #        x = self.history['k']
#
    #    for h, redshift in enumerate(redshifts):
#
    #        iz = np.argmin(np.abs(redshift - self.redshifts))
#
    #        if real_space or (not dimensionless):
    #            norm = 1.
    #        else:
    #            norm = self.history['dTb0_1'][iz]**2
#
    #        # Auto correlations in top row, cross terms on bottom
    #        # z=8 on top, z=12 on bottom
    #        ax = mp.grid[mp.axis_number(include_xcorr, h)]
#
    #        # Plot auto-correlation functions
    #        for i, cf in enumerate(include_fields):
    #            s = '%s_%s' % (prefix, cf)
#
    #            if s not in self.history:
    #                continue
#
    #            if np.all(self.history[s][iz] == 0):
    #                continue
#
    #            x_ch, ps_ch = self._split_cf(redshift, s)
#
    #            if split_by_scale and ('%s_1' % s in self.history.keys()):
    #                x_ch_1, ps_ch_1 = self._split_cf(redshift, s+'_1')
    #                x_ch_2, ps_ch_2 = self._split_cf(redshift, s+'_2')
#
    #            ct = 0
    #            for j, chunk in enumerate(ps_ch):
    #                if np.all(chunk < 0):
    #                    lw = 1
    #                else:
    #                    lw = 3
#
    #                if ct == 0 and (lw == 3 or (j == len(ps_ch) - 1)):
    #                    if real_space:
    #                        label = r'$\xi_{%s}$' % cf
    #                    else:
    #                        label = r'$P_{%s}$' % cf
#
    #                    ct += 1
#
    #                else:
    #                    label = None
#
    #                if real_space:
    #                    mult = 1.
    #                else:
    #                    mult = norm * x_ch[j]**3 / 2. / np.pi**2
#
    #                if 'color' in kwargs:
    #                    c = kwargs['color']
    #                else:
    #                    c = colors[i]
#
    #                if 'ls' in kwargs and (not split_by_scale):
    #                    _ls = kwargs['ls']
    #                else:
    #                    _ls = '-'
#
    #                ax.loglog(x_ch[j], np.abs(chunk) * mult, color=c,
    #                    ls=_ls, alpha=0.5, lw=lw, label=label)
#
#
    #            # Plot one- and two-halo terms separately as dashed/dotted lines
    #            if split_by_scale and ('{}_1'.format(s) in self.history.keys()):
#
    #                ls = ['--', ':']
    #                xlist = [x_ch_1, x_ch_2]
    #                for hh, term in enumerate([ps_ch_1, ps_ch_2]):
    #                    for j, chunk in enumerate(term):
    #                        if np.all(chunk < 0):
    #                            lw = 1
    #                        else:
    #                            lw = 3
#
    #                        if real_space:
    #                            mult = 1.
    #                        else:
    #                            mult = xlist[hh][j]**3 / 2. / np.pi**2
#
    #                        ax.loglog(xlist[hh][j], np.abs(chunk) * mult,
    #                            color=c, ls=ls[hh], alpha=0.5, lw=lw)
#
#
    #        if h == len(redshifts) - 1:
    #            if real_space:
    #                ax.legend(loc='lower left', fontsize=14, ncol=1)
    #            else:
    #                ax.legend(loc='lower right', fontsize=14, ncol=1)
#
#
    #        if real_space:
    #            ax.set_ylim(1e-7, 10)
    #        else:
    #            if dimensionless:
    #                ax.set_ylim(1e-4, 1e5)
    #            else:
    #                ax.set_ylim(1e-7, 1e3)
#
    #        if not gotmp:
    #            ax.annotate(r'$z=%.1f$' % redshift, (0.05, 0.95), xycoords='axes fraction',
    #                ha='left', va='top')
    #            ax.annotate(r'$\bar{Q}_i=%.2f$' % self.history['Qi'][iz], (0.95, 0.95),
    #                xycoords='axes fraction',
    #                ha='right', va='top')
#
    #            ax.annotate(r'$\bar{Q}_h=%.2f$' % self.history['Qh'][iz], (0.95, 0.88),
    #                xycoords='axes fraction',
    #                ha='right', va='top')
#
    #        if not include_xcorr:
    #            continue
#
    #        ax = mp.grid[mp.axis_number(0, h)]
#
    #        # Plot cross-correlations
    #        ls = ':', '--', '-.', '-'
    #        for i, cf in enumerate(['xd', 'cd', 'xco', 'dco']):
#
    #            s = 'cf_%s' % cf
#
    #            if s not in self.history:
    #                continue
#
    #            # Might be multiple sign changes
    #            #data = self.history[s][iz]
    #            #splitter = np.diff(np.sign(data))
    #            #
    #            #if np.all(splitter == 0):
    #            #    chunks = [data]
    #            #    dr_ch = [dr]
    #            #else:
    #            #    splits = np.atleast_1d(np.argwhere(splitter != 0).squeeze()) + 1
    #            #    chunks = np.split(data, splits)
    #            #    dr_ch = np.split(dr, splits)
#
    #            if np.all(self.history[s][iz] == 0):
    #                continue
#
    #            dr_ch, chunks = self._split_cf(redshift, s)
#
    #            for j, chunk in enumerate(chunks):
    #                if np.all(chunk < 0):
    #                    lw = 1
    #                else:
    #                    lw = 3
#
    #                if j == 0:
    #                    if real_space:
    #                        label = r'$\xi_{%s}$' % cf
    #                    else:
    #                        label = r'$P_{%s}$' % cf
#
    #                else:
    #                    label = None
#
    #                if 'color' in kwargs:
    #                    c = kwargs['color']
    #                else:
    #                    c = colors[i]
#
    #                ax.loglog(dr_ch[j], np.abs(chunk), color=c,
    #                    ls='-', alpha=0.5, lw=lw, label=label)
#
    #        if h == len(redshifts) - 1:
    #            ax.legend(loc='lower left', fontsize=14)
#
    #        if real_space:
    #            ax.set_ylim(1e-7, 100)
    #        else:
    #            ax.set_ylim(1e-7, 1e3)
#
    #    if real_space:
    #        mp.grid[mp.upperleft].set_ylabel(r'$\xi_{\mathrm{auto}}$')
#
    #        if include_xcorr:
    #            mp.grid[mp.lowerleft].set_ylabel(r'$\xi_{\mathrm{cross}}$')
    #    else:
    #        mp.grid[mp.upperleft].set_ylabel(labels['dpow'])
#
    #    for i in range(len(redshifts)):
    #        if real_space:
    #            mp.grid[i].set_xlabel(r'$R \ [\mathrm{cMpc}]$')
    #            mp.grid[i].set_xlim(1e-2, 3e3)
    #        else:
    #            mp.grid[i].set_xlabel(r'$k \ [\mathrm{cMpc}^{-1}]$')
    #            mp.grid[i].set_xlim(1e-2, 10)
#
    #    #mp.fix_ticks()
    #    pl.show()
#
    #    return mp

    #def CheckJointProbabilities(self, redshifts, weights=False, mp=None,
    #    mp_kwargs={}):
#
    #    if mp is None:
    #        mp = MultiPanel(dims=(1, len(redshifts)),
    #            padding=(0.25, 0.15), **mp_kwargs)
#
    #    if weights:
    #        all_weights = [(1. - self.history['xibar'])**2, self.history['avg_Ch']**2,
    #            self.history['avg_Cc']**2,
    #            self.history['avg_Ch'] * self.history['avg_Cc']]
    #    else:
    #        all_weights = np.ones(3)
#
    #    for h, redshift in enumerate(redshifts):
#
    #        iz = np.argmin(np.abs(redshift - self.redshifts))
    #        dr = self.history['dr']
#
    #        ax = mp.grid[mp.axis_number(0, h)]
#
    #        # Plot auto-correlation functions
    #        ls = ':', '--', '-.', '-', '-'
    #        colors = ['k'] * 4 + ['b']
    #        for i, cf in enumerate(['ii', 'hh', 'cc', 'hc']):
    #            s = 'jp_%s' % cf
#
    #            if s not in self.history:
    #                continue
#
    #            if np.all(self.history[s][iz] == 0):
    #                continue
#
    #            dr_ch, chunks = self._split_cf(redshift, s)
#
    #            w = all_weights[i]
#
    #            for j, chunk in enumerate(chunks):
    #                if np.all(chunk < 0):
    #                    lw = 1
    #                else:
    #                    lw = 3
#
    #                if j == 0:
    #                    label = r'$P_{%s}$' % cf
    #                else:
    #                    label = None
#
    #                if weights:
    #                    y = np.abs(chunk) * w[iz]
    #                else:
    #                    y = np.abs(chunk)
#
    #                print(cf, redshift, y.shape, dr_ch[j].shape)
    #                ax.loglog(dr_ch[j], y, color=colors[i],
    #                    ls=ls[i], alpha=0.5, lw=lw, label=label)
#
    #        if h == len(redshifts) - 1:
    #            ax.legend(loc='lower left', fontsize=14)
#
    #        ax.annotate(r'$z=%i$' % redshift, (0.05, 0.95), xycoords='axes fraction',
    #            ha='left', va='top')
    #        ax.annotate(r'$\bar{Q}=%.2f$' % self.history['Qi'][iz], (0.95, 0.95),
    #            xycoords='axes fraction',
    #            ha='right', va='top')
#
    #    mp.grid[0].set_ylabel('Joint Probability')
    #    for i in range(len(redshifts)):
    #        mp.grid[i].set_xlabel(r'$R \ [\mathrm{cMpc}]$')
    #        mp.grid[i].set_ylim(1e-8, 10)
#
    #    #mp.fix_ticks()
    #    pl.show()
#
    #    return mp
