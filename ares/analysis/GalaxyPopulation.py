"""

GalaxyPopulation.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Thu Jan 28 12:38:11 PST 2016

Description:

"""

import time
import numpy as np
from ..util import labels
from matplotlib import cm
import matplotlib.pyplot as pl
from .ModelSet import ModelSet
from ..obs.Survey import Survey
from ..obs import DustCorrection
from matplotlib.patches import Patch
from ..util.ReadData import read_lit
from ..util.Aesthetics import labels
from scipy.optimize import curve_fit
import matplotlib.gridspec as gridspec
from ..util.ProgressBar import ProgressBar
from matplotlib.colors import ListedColormap
from ..obs.Photometry import get_filters_from_waves
from ..physics.Constants import rhodot_cgs, cm_per_pc
from ..util.Stats import symmetrize_errors, bin_samples
from ..populations.GalaxyPopulation import GalaxyPopulation as GP
from ..populations.GalaxyEnsemble import GalaxyEnsemble

try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

datasets_lf = ('bouwens2015', 'finkelstein2015', 'bowler2020', 'stefanon2019',
    'mclure2013', 'parsa2016', 'atek2015',  'alavi2016',
    'reddy2009', 'weisz2014', 'bouwens2017', 'oesch2018', 'oesch2013',
    'oesch2014', 'vanderburg2010', 'morishita2018', 'rojasruiz2020')
datasets_smf = ('song2016', 'stefanon2017', 'duncan2014', 'tomczak2014',
	'moustakas2013', 'mortlock2011', 'marchesini2009_10', 'perez2008')
datasets_mzr = ('sanders2015',)
datasets_ssfr = ('dunne2009', 'daddi2007', 'feulner2005', 'kajisawa2010',
	'karim2011', 'noeske2007', 'whitaker2012', 'gonzalez2012')

groups_lf = \
{
 'dropouts': ('parsa2016', 'bouwens2015',
    'finkelstein2015', 'bowler2020','stefanon2019', 'mclure2013',
    'vanderburg2010', 'reddy2009', 'oesch2018', 'oesch2013', 'oesch2014',
    'morishita2018', 'rojasruiz2020'),
 'lensing': ('alavi2016', 'atek2015', 'bouwens2017'),
 'local': ('weisz2014,'),
 'all': datasets_lf,
}
groups_ssfr = {'all': datasets_ssfr}

groups_smf = {'all': datasets_smf}

groups = {'lf': groups_lf, 'smf': groups_smf, 'smf_sf': groups_smf,
    'smf_tot': groups_smf, 'smf_q': groups_smf,
    'mzr': {'all': datasets_mzr}, 'ssfr': groups_ssfr}


colors_cyc = ['m', 'c', 'r', 'g', 'b', 'y', 'orange', 'gray'] * 3
markers = ['o', 's', 'p', 'h', 'D', 'd', '^', 'v', '<', '>'] * 3

default_colors = {}
default_markers = {}
for i, dataset in enumerate(datasets_lf):
    default_colors[dataset] = colors_cyc[i]
    default_markers[dataset] = markers[i]

for i, dataset in enumerate(datasets_smf):
    default_colors[dataset] = colors_cyc[i]
    default_markers[dataset] = markers[i]

for i, dataset in enumerate(datasets_mzr):
    default_colors[dataset] = colors_cyc[i]
    default_markers[dataset] = markers[i]

for i, dataset in enumerate(datasets_ssfr):
    default_colors[dataset] = colors_cyc[i]
    default_markers[dataset] = markers[i]

default_markers['stefanon2017'] = 's'

_ulim_tick = 0.5

class GalaxyPopulation(object): # pragma: no cover
    def __init__(self):
        pass

    def compile_data(self, redshift, sources='all', round_z=False,
        quantity='lf', sources_except=[], just_above=True):
        """
        Create a master dictionary containing the MUV points, phi points,
        and (possibly asymmetric) errorbars for all (or some) data available.

        .. note:: Since we store errorbars in +/- order (if asymmetric), but
            matplotlib.pyplot.errorbar assumes -/+ order, we swap the order here.

        Parameters
        ----------
        z : int, float
            Redshift, dummy!

        """

        data = {}

        if isinstance(sources, basestring):
            if sources in groups[quantity]:
                if sources == 'all':
                    srcs = []
                    for src in groups[quantity]['all']:
                        if src in sources_except:
                            continue
                        srcs.append(src)
                else:
                    srcs = groups[quantity][sources]
            else:
                srcs = [sources]
        else:
            srcs = sources

        for source in srcs:
            src = read_lit(source)

            if redshift not in src.redshifts and (not round_z):
                print("No z={0:g} data in {1!s}.".format(redshift, source))
                continue

            if redshift not in src.redshifts:
                i_close = np.argmin(np.abs(redshift - np.array(src.redshifts)))
                if abs(src.redshifts[i_close] - redshift) <= round_z:
                    z = src.redshifts[i_close]
                else:
                    continue

            else:
                z = redshift

            if quantity not in src.data:
                continue

            data[source] = {}

            if 'label' in src.info:
                data[source]['label'] = src.info['label']

            if quantity in ['lf']:
                data[source]['wavelength'] = src.wavelength

            M = src.data[quantity][z]['M']
            if hasattr(M, 'data'):
                data[source]['M'] = M.data
                mask = M.mask
            else:
                data[source]['M'] = np.array(M)
                mask = np.zeros_like(data[source]['M'])

            if src.units[quantity] == 'log10':
                err_lo = []; err_hi = []; uplims = []; err_mask = []
                for i, err in enumerate(src.data[quantity][z]['err']):


                    if type(err) not in [int, float]:
                        err = np.mean(err)

                    logphi_ML = src.data[quantity][z]['phi'][i]

                    logphi_lo_tmp = logphi_ML - err   # log10 phi
                    logphi_hi_tmp = logphi_ML + err   # log10 phi

                    phi_lo = 10**logphi_lo_tmp
                    phi_hi = 10**logphi_hi_tmp

                    err1 = 10**logphi_ML - phi_lo
                    err2 = phi_hi - 10**logphi_ML

                    if (err < 0):
                        err_hi.append(0.0)
                        err_lo.append(_ulim_tick * 10**logphi_ML)
                    else:
                        err_lo.append(err1)
                        err_hi.append(err2)

                    uplims.append(err < 0)

                    if np.ma.is_masked(err):
                        err_mask.append(True)
                    else:
                        err_mask.append(False)

                data[source]['err'] = (err_lo, err_hi)
                if hasattr(src.data[quantity][z]['phi'], 'data'):
                    data[source]['phi'] = \
                        np.ma.array(10**src.data[quantity][z]['phi'].data,
                            mask=src.data[quantity][z]['phi'].mask)
                else:
                    data[source]['phi'] = \
                        np.ma.array(10**np.array(src.data[quantity][z]['phi'].data),
                            mask=src.data[quantity][z]['phi'].mask)

                data[source]['ulim'] = uplims
            else:

                if hasattr(src.data[quantity][z]['phi'], 'data'):
                    data[source]['phi'] = \
                        np.ma.array(src.data[quantity][z]['phi'].data,
                            mask=src.data[quantity][z]['phi'].mask)
                else:
                    data[source]['phi'] = \
                        np.ma.array(src.data[quantity][z]['phi'].data,
                            mask=src.data[quantity][z]['phi'])

                err_lo = []; err_hi = []; uplims = []; err_mask = []
                for i, err in enumerate(src.data[quantity][z]['err']):

                    if type(err) in [list, tuple, np.ndarray]:
                        err_hi.append(err[0])
                        err_lo.append(err[1])
                        uplims.append(False)
                        err_mask.append(False)
                    elif err is None:
                        err_lo.append(0)
                        err_hi.append(0)
                        uplims.append(False)
                        err_mask.append(True)
                    else:
                        if (err < 0):
                            err_hi.append(0.0)
                            err_lo.append(_ulim_tick * data[source]['phi'][i])
                        else:
                            err_hi.append(err)
                            err_lo.append(err)

                        uplims.append(err < 0)
                        err_mask.append(err < 0)

                data[source]['ulim'] = np.array(uplims)

                err_lo = np.ma.array(err_lo, mask=err_mask)
                err_hi = np.ma.array(err_hi, mask=err_mask)
                mask2 = np.array([err_lo.mask==1, err_hi.mask==1])
                data[source]['err'] = np.ma.array((err_lo, err_hi), mask=mask2)

            data[source]['phi'] = np.ma.array(data[source]['phi'], mask=mask)
            data[source]['M'] = np.ma.array(data[source]['M'], mask=mask)

        return data

    def PlotLF(self, z, ax=None, fig=1, sources='all', round_z=False,
            AUV=None, wavelength=1600., sed_model=None, force_labels=False, **kwargs): # pragma: no cover

        return self.Plot(z=z, ax=ax, fig=fig, sources=sources, round_z=round_z,
            AUV=AUV, wavelength=1600, sed_model=None, quantity='lf',
            force_labels=force_labels, **kwargs)

    def PlotSMF(self, z, ax=None, fig=1, sources='all', round_z=False,
            AUV=None, wavelength=1600., sed_model=None, quantity='smf', force_labels=False, log10Mass=False, **kwargs): # pragma: no cover

        return self.Plot(z=z, ax=ax, fig=fig, sources=sources, round_z=round_z,
            AUV=AUV, wavelength=1600, sed_model=None, quantity=quantity,
            force_labels=force_labels, log10Mass=log10Mass, **kwargs)

    def PlotSSFR(self, z, ax=None, fig=1, sources='all', round_z=False,
            AUV=None, wavelength=1600., sed_model=None, quantity='ssfr', force_labels=False, **kwargs): # pragma: no cover

        return self.Plot(z=z, ax=ax, fig=fig, sources=sources, round_z=round_z,
            AUV=AUV, wavelength=1600, sed_model=None, quantity=quantity,
            force_labels=force_labels, **kwargs)

    def PlotColors(self, pop, axes=None, fig=1, z_uvlf=[4,6,8,10],
        z_beta=[4,5,6,7], z_only=None, sources='all', repeat_z=False, beta_phot=True,
        show_Mstell=True, show_MUV=True, label=None, zcal=None, Mlim=-15,
        dmag=0.5, dMst=0.25, dlam=20, dlam_c94=10, fill=False, extra_pane=False,
        square=False, cmap=None, **kwargs): # pragma: no cover
        """
        Make a nice plot showing UVLF and UV CMD constraints and models.
        """

        num_uvlf_panels = 1
        if type(z_uvlf[0]) not in [int, float, np.int64, np.float64]:
            num_uvlf_panels = 2
            assert not (show_Mstell and show_MUV)
            assert not square
            assert not extra_pane

        if axes is None:

            xp = extra_pane or num_uvlf_panels == 2

            if square:
                dims = (12, 12)
                nrows = 9
                ncols = 4
                hs = 0.1
                ws = 0.8

                assert not xp, "Cannot add extra panel for square plot."
                assert show_Mstell, "No point in square plot if only 2 panels."
            else:
                dims = (24, 6)
                nrows = 4
                ncols = 6 \
                      + 3 * int(num_uvlf_panels == 2) \
                      + 4 * extra_pane \
                      + 2 * (show_Mstell and show_MUV)

                hs = 0.1
                ws = 0.8

            if show_Mstell and show_MUV:
                fig = pl.figure(tight_layout=False, figsize=dims, num=fig)
                fig.subplots_adjust(left=0.1, right=0.9)
                gs = gridspec.GridSpec(nrows, ncols, hspace=hs, wspace=ws,
                    figure=fig)
                ax_extra = None
                xp = 0
            else:
                fig = pl.figure(tight_layout=False, figsize=(12+xp*6, 6),
                    num=fig)
                fig.subplots_adjust(left=0.1 ,right=0.9)
                # nrows, ncols
                gs = gridspec.GridSpec(nrows, ncols, hspace=0.0, wspace=0.05,
                    figure=fig)

            s = int(square)

            if show_Mstell and show_MUV:
                ax_uvlf = fig.add_subplot(gs[:4,0:2])

                ax_cmr4 = fig.add_subplot(gs[0,2:4])
                ax_cmr6 = fig.add_subplot(gs[1,2:4])
                ax_cmr8 = fig.add_subplot(gs[2,2:4])
                ax_cmr10 = fig.add_subplot(gs[3,2:4])

                ax_smf = fig.add_subplot(gs[s*5:,(1-s)*4:(1-s)*4+2])
                ax_cMs4 = fig.add_subplot(gs[s*5+0, (1-s)*4+2:])
                ax_cMs6 = fig.add_subplot(gs[s*5+1, (1-s)*4+2:])
                ax_cMs8 = fig.add_subplot(gs[s*5+2, (1-s)*4+2:])
                ax_cMs10 = fig.add_subplot(gs[s*5+3,(1-s)*4+2:])

                ax_cMs = [ax_cMs4, ax_cMs6, ax_cMs8, ax_cMs10]
            else:
                if xp and num_uvlf_panels == 1:
                    # cols, rows
                    ax_extra = fig.add_subplot(gs[:,0:3])
                else:
                    ax_extra = None

                if num_uvlf_panels == 2:
                    ax_uvlf = fig.add_subplot(gs[:,0:3])
                    ax_uvlf2 = fig.add_subplot(gs[:,3:6])
                    ax_cmr4 = fig.add_subplot(gs[0,6:])
                    ax_cmr6 = fig.add_subplot(gs[1,6:])
                    ax_cmr8 = fig.add_subplot(gs[2,6:])
                    ax_cmr10 = fig.add_subplot(gs[3,6:])
                else:
                    ax_uvlf = fig.add_subplot(gs[:,0+4*xp:4*xp+3])
                    ax_uvlf2 = None

                    ax_cmr4 = fig.add_subplot(gs[0,3+4*xp:])
                    ax_cmr6 = fig.add_subplot(gs[1,3+4*xp:])
                    ax_cmr8 = fig.add_subplot(gs[2,3+4*xp:])
                    ax_cmr10 = fig.add_subplot(gs[3,3+4*xp:])

                if show_Mstell and (not show_MUV):
                    ax_smf = ax_uvlf
                    ax_smf2 = ax_uvlf2
                    ax_cMs4 = ax_cmr4
                    ax_cMs6 = ax_cmr6
                    ax_cMs8 = ax_cmr8
                    ax_cMs10 = ax_cmr10
                    ax_cMs = [ax_cMs4, ax_cMs6, ax_cMs8, ax_cMs10]
                else:
                    ax_cMs = []
                    ax_smf = None

            ax_cmd = [ax_cmr4, ax_cmr6, ax_cmr8, ax_cmr10]

            axes = ax_uvlf, ax_cmd, ax_smf, ax_cMs, ax_extra

            had_axes = False

        else:
            had_axes = True
            ax_uvlf, ax_cmd, ax_smf, ax_cMs, ax_extra = axes
            ax_cmr4, ax_cmr6, ax_cmr8, ax_cmr10 = ax_cmd

            if num_uvlf_panels == 2 and show_MUV:
                ax_uvlf2 = ax_extra
            if num_uvlf_panels == 2 and show_Mstell:
                ax_smf2 = ax_extra

            if show_Mstell:
                ax_cMs4, ax_cMs6, ax_cMs8, ax_cMs10 = ax_cMs

        if type(pop) in [list, tuple]:
            pops = pop
        else:
            pops = [pop]

        if zcal is not None:
            if type(zcal) != list:
                zcal = [zcal]

        l11 = read_lit('lee2011')
        b14 = read_lit('bouwens2014')
        f12 = read_lit('finkelstein2012')

        _colors = {4: 'k', 5: 'r', 6: 'b', 7: 'y', 8: 'c', 9: 'g', 10: 'm'}

        if num_uvlf_panels == 2:
            z_uvlf_flat = []
            for element in z_uvlf:
                z_uvlf_flat.extend(element)
        else:
            z_uvlf_flat = z_uvlf

        zall = np.sort(np.unique(np.concatenate((z_uvlf_flat, z_beta))))

        if cmap is not None:

            if (type(cmap) is str) or isinstance(cmap, ListedColormap):
                dz = 0.05
                _zall = np.arange(zall.min()-0.25, zall.max()+0.25, dz)
                znormed = (_zall - _zall[0]) / float(_zall[-1] - _zall[0])
                ch = cm.get_cmap(cmap, _zall.size)
                normz = lambda zz: (zz - _zall[0]) / float(_zall[-1] - _zall[0])
                colors = lambda z: ch(normz(z))
                self.colors = colors
            else:
                colors = cmap
        else:
            colors = lambda z: _colors[int(z)]

        ##
        # Plot data
        ##
        mkw = {'capthick': 1, 'elinewidth': 1, 'alpha': 1.0, 'capsize': 1}

        ct_lf = np.zeros(num_uvlf_panels)
        ct_b = 0
        for j, z in enumerate(zall):

            zstr = round(z)
            if z_only is not None:
                if zstr != z_only:
                    continue

            zint = int(round(z, 0))

            if z in z_uvlf_flat:

                if num_uvlf_panels == 2:
                    if z in z_uvlf[0]:
                        _ax = ax_uvlf
                        k = 0
                    else:
                        if show_MUV:
                            _ax = ax_uvlf2
                        else:
                            _ax = ax_smf2

                        k = 1
                else:
                    _ax = ax_uvlf
                    k = 0


                _ax_ = self.PlotLF(z, ax=_ax, color=colors(zint), mfc=colors(zint),
                    mec=colors(zint), sources=sources, round_z=0.23,
                    use_labels=0)

                if show_MUV and (not had_axes):
                    if zcal is not None and z in zcal:
                        bbox = dict(facecolor='none', edgecolor=colors(zint), fc='w',
                            boxstyle='round,pad=0.3', alpha=1., zorder=1000)
                    else:
                        bbox = None

                    _ax.text(0.95, 0.4-0.1*ct_lf[k], r'$z \sim {}$'.format(z),
                        transform=_ax.transAxes, color=colors(zint),
                        ha='right', va='top', bbox=bbox, fontsize=20)

                    #ax_uvlf.annotate(r'$z \sim {}$'.format(z), (0.95, 0.25-0.05*ct_lf),
                    #    xycoords='axes fraction', color=colors[z], ha='right', va='top')


                if show_Mstell:

                    if (not show_MUV):
                        _ax2 = _ax
                    else:
                        _ax2 = ax_smf

                    _ax_ = self.PlotSMF(z, ax=_ax2, color=colors(zint), mfc=colors(zint),
                        mec=colors(zint), sources=sources, round_z=0.21, use_labels=0)

                    if not had_axes:
                        _ax2.annotate(r'$z \sim {}$'.format(zint), (0.05, 0.4-0.1*ct_lf[k]),
                            xycoords='axes fraction', color=colors(zint),
                            ha='left', va='top', fontsize=20)

                ct_lf[k] += 1

            if z not in z_beta:
                continue

            kcmd = np.argmin(np.abs(z - np.array(z_beta)))

            if zint in b14.data['beta'] and show_MUV:
                err = b14.data['beta'][zint]['err'] + b14.data['beta'][zint]['sys']
                ax_cmd[kcmd].errorbar(b14.data['beta'][zint]['M'], b14.data['beta'][zint]['beta'],
                    yerr=err,
                    fmt='o', color=colors(zint), label=b14.info['label'] if j == 0 else None,
                    **mkw)

            #if z in l11.data['beta']:
            #    ax_cmd[j].errorbar(l11.data['beta'][z]['M'], l11.data['beta'][z]['beta'],
            #        l11.data['beta'][z]['err'],
            #        fmt='*', color=colors[z], label=r'Lee+ 2011' if j == 0 else None,
            #        **mkw)

            if not had_axes:

                if zcal is not None and z in zcal:
                    bbox= dict(facecolor='none', edgecolor=colors(zint), fc='w',
                        boxstyle='round,pad=0.3', alpha=1., zorder=1000)
                else:
                    bbox = None

                if show_MUV:
                    ax_cmd[kcmd].text(0.05, 0.05, r'$z \sim {}$'.format(zint),
                        transform=ax_cmd[kcmd].transAxes, color=colors(zint),
                        ha='left', va='bottom', bbox=bbox, fontsize=20)

                #ax_cmd[j].annotate(r'$z \sim {}$'.format(z), (0.95, 0.95),
                #    ha='right', va='top', xycoords='axes fraction', color=colors[z])

            ct_b += 1

            if not show_Mstell:
                continue

            if z in f12.data['beta']:
                err = f12.data['beta'][zint]['err']
                ax_cMs[kcmd].errorbar(10**f12.data['beta'][zint]['Ms'],
                    f12.data['beta'][zint]['beta'], err.T[-1::-1],
                    fmt='o', color=colors(zint),
                    label=f12.info['label'] if j == 0 else None,
                    **mkw)

        ##
        # Plot models
        ##
        Ms = np.arange(2, 13.+dMst, dMst)
        mags = np.arange(-30, 10-dmag, dmag)
        mags_cr = np.arange(-25, -10, dmag)
        hst_shallow = b14.filt_shallow
        hst_deep = b14.filt_deep
        calzetti = read_lit('calzetti1994').windows

        uvlf_by_pop = {}
        smf_by_pop = {}
        bphot_by_pop = {}
        bc94_by_pop = {}
        for h, pop in enumerate(pops):

            uvlf_by_pop[h] = {}
            smf_by_pop[h] = {}
            bphot_by_pop[h] = {}
            bc94_by_pop[h] = {}

            for j, z in enumerate(zall):
                zstr = round(z)

                if z_only is not None:
                    if zstr != z_only:
                        continue

                zint = int(round(z, 0))

                if z in z_uvlf_flat:

                    if num_uvlf_panels == 2:
                        if z in z_uvlf[0]:
                            _ax = ax_uvlf
                        else:
                            if show_MUV:
                                _ax = ax_uvlf2
                            else:
                                _ax = ax_smf2
                    else:
                        _ax = ax_uvlf

                    if show_MUV:
                        _mags, phi = pop.get_lf(z, mags)
                        uvlf_by_pop[h][z] = phi

                        if not fill:
                            _ax.semilogy(mags, phi, color=colors(zint),
                                label=label if j == 0 else None, **kwargs)

                    if show_Mstell:

                        if (not show_MUV):
                            _ax2 = _ax
                        else:
                            _ax2 = ax_smf

                        _bins, phi = pop.StellarMassFunction(z, bins=Ms)
                        smf_by_pop[h][z] = phi

                        if not fill:
                            _ax2.semilogy(10**Ms, phi,
                                color=colors(zint),
                                label=label if j == 0 else None,**kwargs)

                if z not in z_beta:
                    continue

                kcmd = np.argmin(np.abs(z - np.array(z_beta)))

                if zstr >= 7:
                    hst_filt = hst_deep
                else:
                    hst_filt = hst_shallow

                presets = 'hst' if zstr <= 8 else 'jwst-m'

                if beta_phot:
                    beta = pop.Beta(z, Mbins=mags_cr, return_binned=True,
                        presets=presets, rest_wave=None,
                        dlam=dlam)

                else:
                    beta = pop.Beta(z, Mbins=mags_cr, return_binned=True,
                        rest_wave=(1600., 3000.), dlam=dlam)

                bphot_by_pop[h][z] = beta

                # Mask
                ok = np.logical_and(np.isfinite(beta), beta > -99999)
                if not fill and ok.sum() > 0:
                    ax_cmd[kcmd].plot(mags_cr[ok==1], beta[ok==1], color=colors(zint), **kwargs)

                if show_Mstell:

                    _beta_c94 = pop.Beta(z, return_binned=False,
                        cam='calzetti', filters=calzetti, dlam=dlam_c94,
                        rest_wave=None)

                    # _beta_c94 is 'raw', i.e., unbinned UV slopes for all halos.
                    # Just need to bin as function of stellar mass.
                    _Ms = pop.get_field(z, 'Ms')
                    _nh = pop.get_field(z, 'nh')
                    _x, _b, _err, _N = bin_samples(np.log10(_Ms), _beta_c94, Ms,
                        weights=_nh)

                    bc94_by_pop[h][z] = _b

                    if not fill:
                        ax_cMs[kcmd].plot(10**_x, _b, color=colors(zint), **kwargs)

                    ax_cMs[kcmd].annotate(r'$z \sim {}$'.format(zint), (0.05, 0.95),
                        ha='left', va='top', xycoords='axes fraction',
                        color=colors(zint), fontsize=20)

                if repeat_z and (j == 0) and (not fill):
                    for k in range(1, 4):
                        ax_cmd[k].plot(mags_cr, beta, color=colors(zint), **kwargs)
                        if show_Mstell:
                            ax_cMs[k].plot(10**Ms, _b, color=colors(zint), **kwargs)

        ##
        # Plot filled contours under certain conditions
        if fill and len(pops) == 2:
            for j, z in enumerate(zall):
                zstr = round(z)

                if z_only is not None:
                    if zstr != z_only:
                        continue

                if z in z_uvlf_flat:
                    if num_uvlf_panels == 2:
                        if z in z_uvlf[0]:
                            _ax = ax_uvlf
                        else:
                            if show_MUV:
                                _ax = ax_uvlf2
                            else:
                                _ax = ax_smf2
                    else:
                        _ax = ax_uvlf

                    if show_MUV:
                        _ax.fill_between(mags, uvlf_by_pop[0][z],
                            uvlf_by_pop[1][z], color=colors(z),
                            label=label if j == 0 else None, **kwargs)

                    if show_Mstell:
                        ax_smf.fill_between(10**Ms, smf_by_pop[0][z],
                            smf_by_pop[1][z], color=colors(zint), **kwargs)

                if z not in z_beta:
                    continue

                #ok = np.logical_and(np.isfinite(beta), beta > -99999)

                kcmd = np.argmin(np.abs(z - np.array(z_beta)))

                ax_cmd[kcmd].fill_between(mags_cr, bphot_by_pop[0][z],
                    bphot_by_pop[1][z], color=colors(zint), **kwargs)

                if show_Mstell:
                    ax_cMs[kcmd].fill_between(10**Ms, bc94_by_pop[0][z],
                        bc94_by_pop[1][z], color=colors(zint), **kwargs)

        ##
        # Clean-up
        ##
        if num_uvlf_panels == 2:
            if show_MUV:
                ax_extra = ax_uvlf2
            else:
                ax_extra = ax_smf2

        if show_MUV:
            _axes_uvlf = [ax_uvlf] if num_uvlf_panels == 1 else [ax_uvlf, ax_uvlf2]
        else:
            _axes_uvlf = [ax_uvlf] if num_uvlf_panels == 1 else [ax_uvlf, ax_smf2]

        for i, ax in enumerate(_axes_uvlf + ax_cmd):

            if not show_MUV:
                break

            if ax is None:
                continue

            ax.set_xlim(-24, Mlim)
            ax.set_xticks(np.arange(-24, Mlim, 2))
            ax.set_xticks(np.arange(-24, Mlim, 1), minor=True)

            if i > (num_uvlf_panels - 1):
                if show_MUV:
                    ax.set_ylabel(r'$\beta_{\mathrm{hst}}$')

                ax.set_yticks(np.arange(-2.8, -0.8, 0.4))
                ax.set_yticks(np.arange(-2.9, -1., 0.1), minor=True)
                ax.set_ylim(-2.9, -1.)

                if not show_Mstell:
                    ax.yaxis.tick_right()
                    ax.yaxis.set_label_position("right")

                if i < 4 + (num_uvlf_panels - 1):
                    ax.set_xticklabels([])
                else:
                    if beta_phot:
                        ax.set_xlabel(r'$\langle M_{\mathrm{UV}} \rangle$')
                    else:
                        ax.set_xlabel(r'$M_{\mathrm{UV}}$')

                ax.yaxis.set_ticks_position('both')
            else:
                ax.set_xlabel(r'$M_{1600}$')
                ax.set_ylim(1e-7, 1e-1)
                if i == 0:
                    ax.set_ylabel(labels['galaxy_lf'])
                elif num_uvlf_panels == 2 and i == 1:
                    ax.set_yticklabels([])

        if show_Mstell:
            ax_smf.set_xlabel(r'$M_{\ast} / M_{\odot}$')
            ax_smf.set_ylabel(labels['galaxy_smf'])
            ax_smf.set_xscale('log')
            ax_smf.set_ylim(1e-7, 1e-1)
            ax_smf.set_xlim(1e7, 7e11)

            if num_uvlf_panels == 2:
                ax_smf2.set_xlabel(r'$M_{\ast} / M_{\odot}$')
                ax_smf2.set_xscale('log')
                ax_smf2.set_ylim(1e-7, 1e-1)
                ax_smf2.set_xlim(1e7, 7e11)
                ax_smf2.set_yticklabels([])

            for i, ax in enumerate([ax_cMs4, ax_cMs6, ax_cMs8, ax_cMs10]):

                if ax is None:
                    continue

                ax.set_xscale('log')
                ax.set_xlim(1e7, 7e11)
                ax.set_ylabel(r'$\beta_{\mathrm{c94}}$')
                ax.set_yticks(np.arange(-2.8, -0.8, 0.4))
                ax.set_yticks(np.arange(-2.9, -1., 0.1), minor=True)
                ax.set_ylim(-2.9, -1.)

                if not show_MUV:
                    ax.yaxis.set_label_position("right")
                    ax.yaxis.set_ticks_position('right')

                if i < 3:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel(r'$M_{\ast} / M_{\odot}$')

        return ax_uvlf, ax_cmd, ax_smf, ax_cMs, ax_extra

    def PlotColorEvolution(self, pop, zarr=None, axes=None, fig=1,
        wave_lo=None, wave_hi=None, show_beta_spec=True, dlam=1,
        show_beta_hst=True, show_beta_jwst_W=True, show_beta_jwst_M=True,
        magmethod='gmean', include_Mstell=True, MUV=[-19.5], ls='-',
        colors=['r'],
        return_data=True, data=None, augment_filters=True, **kwargs):
        """
        Plot Beta(z) at fixed MUV and (optionally) Mstell.
        """

        if axes is None:
            fig = pl.figure(tight_layout=False, figsize=(8, 8), num=fig)
            fig.subplots_adjust(left=0.2)
            gs = gridspec.GridSpec(2, 1+include_Mstell, hspace=0.05,
                wspace=0.05, figure=fig)

            axB = fig.add_subplot(gs[0,0])
            axD = fig.add_subplot(gs[1,0])

            if include_Mstell:
                axB2 = fig.add_subplot(gs[0,1])
                axD2 = fig.add_subplot(gs[1,1])
            else:
                axB2 = axD2 = None
        else:
            if include_Mstell:
                axB, axD, axB2, axD2 = axes
            else:
                axB, axD = axes
                axB2 = axD2 = None


        assert len(colors) == len(MUV), \
            "`MUV` and `colors` must be of same length."

        # Plot the Bouwens data
        zbrack = [3.8, 5.0, 5.9, 7.0, 8.0]
        Beta195 = [-1.85, -1.91, -2.00, -2.05, -2.13]
        Beta195_err = [0.01, 0.02, 0.05, 0.09, 0.44]
        Beta195_sys = [0.06, 0.06, 0.08, 0.13, 0.27]

        dBdMUV = [-0.11, -0.14, -0.2, -0.2]
        dB_err = [0.01, 0.02, 0.04, 0.07]

        axB.errorbar(zbrack, Beta195, yerr=Beta195_sys, fmt='o', zorder=10,
            color='r')
        axD.errorbar(zbrack[:-1], -np.array(dBdMUV), yerr=np.array(dB_err),
            fmt='o', zorder=10, color='r')

        mags = np.arange(-25, -10, 0.1)
        mags_cr = np.arange(-25.5, -10, 0.5)

        if zarr is None:
            zarr = np.arange(4, 12., 1.)

        linfunc = lambda x, p0, p1: p0 * (x - 8.) + p1
        cubfunc = lambda x, p0, p1, p2: p0 * (x - 8.)**2 + p1 * (x - 8.) + p2

        f12 = read_lit('finkelstein2012')
        calzetti = read_lit('calzetti1994').windows

        if wave_lo is None:
            wave_lo = np.min(calzetti)
        if wave_hi is None:
            wave_hi = np.max(calzetti)

        ##
        # Plot data: use same color-conventions as F12 for Mstell-beta stuff.
        ##
        if include_Mstell:
            markers = 'v', 's', '^'
            Mstell = np.array([7.5, 8.5, 9.5])
            for z in [4,5,6,7,8]:
                for i, _Mstell in enumerate(Mstell):
                    x = z
                    y = f12.data['beta'][z]['beta'][i]
                    yerr = np.array([f12.data['beta'][z]['err'][i]]).T[-1::-1]

                    if axes is None:
                        lab = r'$%.1f \leq \log_{10} (M_{\ast} / M_{\odot}) \leq %.1f$' \
                            % (_Mstell-0.5, _Mstell+0.5) if z == 4 else None
                    else:
                        lab = None

                    axB2.errorbar(z, y, yerr=yerr, fmt=markers[i],
                        color=colors[-1::-1][i], alpha=1.)

                    #if z == 4:
                    #    axB2.annotate(lab, (0.95, 0.95-0.05*i), ha='right', va='top',
                    #        color=colors[i], fontsize=12, xycoords='axes fraction')

                axD2.errorbar(z, f12.data['slope_wrt_mass'][z]['slope'],
                    yerr=f12.data['slope_wrt_mass'][z]['err'],
                    color='k', fmt='o', alpha=1.)

        ##
        # Continue with model predictions
        ##

        # For CANDELS, ERS
        b14 = read_lit('bouwens2014')
        hst_shallow = b14.filt_shallow
        hst_deep = b14.filt_deep

        if (show_beta_jwst_W or show_beta_jwst_M):
            nircam = Survey(cam='nircam')
            nircam_M = nircam._read_nircam(filter_set='M')
            nircam_W = nircam._read_nircam(filter_set='W')

        ##
        # Loop over models and reconstruct best-fitting Beta(z).
        ##
        Ms_b = np.arange(6.5, 11., 0.5)

        if len(MUV) != len(ls):
            ls = ['-'] * len(MUV)

        ##
        # Won't be able to do DerivedBlob for 'nozevo' case because we only
        # saved at one redshift :( Will be crude for others. Could re-generate
        # samples later (parallelize, on cluster).
        ##
        mkw = {'capthick': 1, 'elinewidth': 1, 'alpha': 1.0, 'capsize': 1}

        pb = ProgressBar(zarr.size, name='beta(z)')
        pb.start()

        B195_hst       = -99999 * np.ones((len(zarr), len(MUV)))
        dBdM195_hst    = -99999 * np.ones((len(zarr), len(MUV)))
        B195_spec      = -99999 * np.ones((len(zarr), len(MUV)))
        dBdM195_spec   = -99999 * np.ones((len(zarr), len(MUV)))
        B195_jwst      = -99999 * np.ones((len(zarr), len(MUV)))
        dBdM195_jwst   = -99999 * np.ones((len(zarr), len(MUV)))
        B195_M         = -99999 * np.ones((len(zarr), len(MUV)))
        dBdM195_M      = -99999 * np.ones((len(zarr), len(MUV)))
        BMstell        = -99999 * np.ones((len(zarr), len(Ms_b)))
        dBMstell       = -99999 * np.ones((len(zarr), len(Ms_b)))
        for j, z in enumerate(zarr):

            if data is not None:
                break

            t1 = time.time()
            print("Colors at z={}...".format(z))

            zstr = round(z)

            if zstr >= 6:
                hst_filt = hst_deep
            else:
                hst_filt = hst_shallow

            fset = None
            if zstr <= 8:
                cam = ('wfc', 'wfc3')
                filt = hst_filt[zstr]

                beta_hst = pop.Beta(z, Mbins=mags_cr, return_binned=True,
                    cam=cam, filters=filt, filter_set=fset, rest_wave=None,
                    magmethod=magmethod)
               #beta_hst_M1600 = pop.Beta(z, Mbins=mags_cr, return_binned=True,
               #    cam=cam, filters=filt, filter_set=fset, rest_wave=None,
               #    magmethod='mono')
            else:
                beta_hst = beta_hst_M1600 = -np.inf * np.ones_like(mags_cr)

            # Fit smooth function to Beta(MUV), use to interpolate
            # and compute derivatives.

            # First up: HST Beta.
            if z <= 8:
                slope, func, func_p = pop.dBeta_dMUV(z, magbins=mags_cr,
                    return_funcs=True, model='quad3', presets='hst',
                    maglim=(-22.5, -16.5))

                # Compute beta and dBeta/dMUV at a few magnitudes
                if func is not None:
                    B195_hst[j,:] = func(MUV)
                    dBdM195_hst[j,:] = func_p(MUV)
                else:
                    print("# WARNING: z={} dBeta_dMUV yielded fully-masked dataset.".format(z))

            ##
            # Slope of UV slope in Calzetti windows vs. magnitude
            slope, func, func_p = pop.dBeta_dMUV(z, magbins=mags_cr,
                magmethod='mono',
                return_funcs=True, model='quad3', presets='calzetti', dlam=1.,
                maglim=(-22.5, -16.5))

            if func is not None:
                B195_spec[j,:] = func(MUV)
                dBdM195_spec[j,:] = func_p(MUV)
            else:
                print("# WARNING: z={} dBeta_dMUV yielded fully-masked dataset.".format(z))

            # Compute beta(Mstell)
            if include_Mstell:
                beta_Mst = pop.Beta(z, Mwave=1600., return_binned=False,
                    cam='calzetti', filters=calzetti, dlam=dlam, rest_wave=None,
                    Mstell=10**Ms_b, massbins=Ms_b)

                dbeta_Mst, func, func_p = pop.dBeta_dMstell(z, Mstell=10**Ms_b,
                    massbins=Ms_b, dlam=1., return_funcs=True)

                BMstell[j,:] = func(Ms_b)
                dBMstell[j,:] = func_p(Ms_b)

            # Compute beta given JWST W only
            #
            if (show_beta_jwst_W or show_beta_jwst_M) and z >= 4:

                if show_beta_jwst_W:
                    nircam_W_fil = get_filters_from_waves(z, nircam_W, wave_lo,
                        wave_hi)
                    # Extend the wavelength range until we get two filters

                    if augment_filters:

                        ct = 1
                        while len(nircam_W_fil) < 2:
                            nircam_W_fil = get_filters_from_waves(z, nircam_W,
                                wave_lo, wave_hi + 10 * ct)

                            ct += 1

                        if ct > 1:
                            print("For JWST W filters at z={}, extend wave_hi to {}A".format(z,
                                wave_hi + 10 * (ct - 1)))


                    filt2 = tuple(nircam_W_fil)

                    beta_W = pop.Beta(z, Mbins=mags_cr, return_binned=True,
                        cam=('nircam', ), filters=filt2, filter_set=fset,
                        rest_wave=None, magmethod=magmethod)

                    slope, func, func_p = pop.dBeta_dMUV(z, magbins=mags_cr,
                        return_funcs=True, model='quad3', presets='nircam-w',
                        maglim=(-22.5, -16.5))

                    if func is not None:
                        # Compute beta and dBeta/dMUV at a few magnitudes
                        #for l, mag in enumerate(MUV):
                        B195_jwst[j,:] = func(MUV)
                        dBdM195_jwst[j,:] = func_p(MUV)

                # Compute beta w/ JWST 'M' only
                if show_beta_jwst_M:
                    nircam_M_fil = get_filters_from_waves(z, nircam_M, wave_lo,
                        wave_hi)

                    if augment_filters:

                        ct = 1
                        while len(nircam_M_fil) < 2:
                            nircam_M_fil = get_filters_from_waves(z, nircam_M,
                                wave_lo, wave_hi + 10 * ct)

                            ct += 1

                        if ct > 1:
                            print("For JWST M filters at z={}, extend wave_hi to {}A".format(z,
                                wave_hi + 10 * (ct - 1)))

                    filt3 = tuple(nircam_M_fil)

                    if (z >= 4 and augment_filters) or (z >= 6 and not augment_filters):
                        #beta_M = pop.Beta(z, Mbins=mags_cr, return_binned=True,
                        #    cam=('nircam',), filters=filt3, rest_wave=None,
                        #    magmethod=magmethod)

                        beta_M = pop.Beta(z, Mbins=mags_cr, return_binned=True,
                            presets='jwst', rest_wave=None,
                            magmethod=magmethod)

                        slope, func, func_p = pop.dBeta_dMUV(z, magbins=mags_cr,
                            return_funcs=True, model='quad3', presets='nircam-m',
                            maglim=(-22.5, -16.5))

                        if func is not None:
                            # Compute beta and dBeta/dMUV at a few magnitudes
                            B195_M[j,:] = func(MUV)
                            dBdM195_M[j,:] = func_p(MUV)

                    else:
                        beta_M = -np.inf * np.ones_like(mags_cr)
            else:
                beta_W = beta_M = None

            pb.update(j)
            t2 = time.time()

            print(t2 - t1)

        pb.finish()

        if data is not None:
            _MUV, B195_spec, B195_hst, B195_jwst, B195_M, BMstell, \
                dBdM195_spec, dBdM195_hst, dBdM195_jwst, dBdM195_M, \
                dBMstell = data

            assert np.array_equal(_MUV, MUV)

        ##
        # Finish up and plot.
        ##
        if show_beta_spec:
            for l, mag in enumerate(MUV):
                _beta = B195_spec[:,l]
                ok = _beta > -99999

                axB.plot(zarr[ok==1], _beta[ok==1], lw=1,
                    color=colors[l], ls=':', label='c94' if l==0 else None)
                axD.plot(zarr[ok==1], -dBdM195_spec[ok==1,l], lw=1,
                    color=colors[l], ls=':')

        if show_beta_hst:
            for l, mag in enumerate(MUV):
                _beta = B195_hst[:,l]
                ok = _beta > -99999
                axB.plot(zarr[ok==1], _beta[ok==1], lw=2,
                    color=colors[l], ls='-', label='hst' if l==0 else None)
                axD.plot(zarr[ok==1], -dBdM195_hst[ok==1,l], lw=2,
                    color=colors[l], ls='-')

        if show_beta_jwst_W:
            for l, mag in enumerate(MUV):
                _beta = B195_jwst[:,l]
                ok = _beta > -99999
                #ok = np.logical_and(_beta > -99999, zarr <= 9.)
                axB.plot(zarr[ok==1], _beta[ok==1], lw=2,
                    color=colors[l], ls='-.', label='jwst-W' if l==0 else None)
                axD.plot(zarr[ok==1], -dBdM195_jwst[ok==1,l], lw=2,
                    color=colors[l], ls='-.')

        if show_beta_jwst_M:
            for l, mag in enumerate(MUV):
                _beta = B195_M[:,l]
                ok = _beta > -99999
                ok = np.logical_and(ok, zarr >= 6)
                axB.plot(zarr[ok==1], _beta[ok==1], lw=2,
                    color=colors[l], ls='--', label='jwst-M' if l==0 else None)
                axD.plot(zarr[ok==1], -dBdM195_M[ok==1,l], lw=2,
                    color=colors[l], ls='--')

        ##
        # Plot Mstell stuff
        ##
        if include_Mstell:
            _ls = '-', '--', ':', '-.'
            for _j, logM in enumerate([9.5, 8.5, 7.5]):
                j = np.argmin(np.abs(Ms_b - logM))
                axB2.plot(zarr, BMstell[:,j], ls=':', color=colors[_j],
                    label=r'$M_{\ast} = 10^{%i} \ M_{\odot}$' % logM)
                axD2.plot(zarr, dBMstell[:,j], ls=':', color=colors[_j],
                    label=r'$M_{\ast} = 10^{%i} \ M_{\odot}$' % logM)


        ##
        # Clean up
        ##
        axD.set_yticks(np.arange(0.0, 0.6, 0.2))
        axD.set_yticks(np.arange(0.0, 0.6, 0.1), minor=True)
        #axD.legend(loc='upper right', frameon=True, fontsize=8,
        #    handlelength=2, ncol=1)
        axD.set_xlim(3.5, zarr.max()+0.5)
        axD.set_ylim(-0.05, 0.5)
        axD.yaxis.set_ticks_position('both')

        axB.set_xlim(3.5, zarr.max()+0.5)
        axB.set_ylim(-3.05, -1.3)
        axB.yaxis.set_ticks_position('both')
        axB.set_yticks(np.arange(-3, -1.3, 0.25), minor=False)
        axB.set_yticks(np.arange(-3, -1.3, 0.05), minor=True)
        axB.legend(loc='upper right', frameon=True, fontsize=8,
            handlelength=4, ncol=1)
        axB.set_ylim(-2.8, -1.4)
        axB.set_xticklabels([])

        if include_Mstell:
            axB2.set_ylim(-3.05, -1.3)
            axB2.set_xlim(3.5, zarr.max()+0.5)
            axB2.set_yticks(np.arange(-3, -1.3, 0.25), minor=False)
            axB2.set_yticks(np.arange(-3, -1.3, 0.05), minor=True)
            axB2.legend(loc='upper right', frameon=True, fontsize=8,
                handlelength=2, ncol=1)

            axD2.set_xlim(3.5, zarr.max()+0.5)
            axD2.set_yticks(np.arange(0.0, 0.6, 0.2))
            axD2.set_yticks(np.arange(0.0, 0.6, 0.1), minor=True)
            axD2.set_ylim(-0.05, 0.5)
            axD2.legend(loc='upper right', frameon=True, fontsize=8,
                handlelength=2, ncol=1)

            if axes is None:
                axD2.set_xlabel(r'$z$')
                #axB2.set_xlabel(r'$z$')
                axB2.set_xticklabels([])
                axB2.yaxis.tick_right()
                axD2.yaxis.tick_right()
                axB2.yaxis.set_ticks_position('both')
                axD2.yaxis.set_ticks_position('both')
                axB2.yaxis.set_label_position("right")
                axD2.yaxis.set_label_position("right")
                axB2.set_ylabel(r'$\beta_{\mathrm{c94}}$')
                axD2.set_ylabel(r'$d\beta_{\mathrm{c94}}/dlog_{10}M_{\ast}$')
                axB2.set_ylim(-2.8, -1.4)


        if axes is None:
            axB.set_ylabel(r'$\beta$')
            axD.set_ylabel(r'$-d\beta/dM_{\mathrm{UV}}$')
            axD.set_xlabel(r'$z$')
            #axB.set_xlabel(r'$z$')

        #for ax in [axB, axD, axB2, axD2]:
        #    if ax is None:
        #        continue
        #    ax.yaxis.set_label_coords(-0.1-0.08*include_Mstell, 0.5)
        #    ax.yaxis.set_label_coords(-0.1-0.08*include_Mstell, 0.5)

        if return_data:
            data = (MUV, B195_spec, B195_hst, B195_jwst, B195_M, BMstell,
                dBdM195_spec, dBdM195_hst, dBdM195_jwst, dBdM195_M, dBMstell)

            return (axB, axD, axB2, axD2), data
        else:
            data = None

            return axB, axD, axB2, axD2

    def Plot(self, z, ax=None, fig=1, sources='all', round_z=False,
        force_labels=False, AUV=None, wavelength=1600., sed_model=None,
        quantity='lf', use_labels=True,
        take_log=False, imf=None, mags='intrinsic', sources_except=[],
        log10Mass=False, **kwargs): # pragma: no cover
        """
        Plot the luminosity function data at a given redshift.

        Parameters
        ----------
        z : int, float
            Redshift of interest
        wavelength : int, float
            Wavelength (in Angstroms) of LF.
        sed_model : instance
            ares.sources.SynthesisModel
        imf : str
            Stellar initial mass function. Will be used to convert stellar
            masses, if supplied.

        """

        if ax is None:
            gotax = False
            fig = pl.figure(fig)
            ax = fig.add_subplot(111)
        else:
            gotax = True

        data = self.compile_data(z, sources, round_z=round_z,
            quantity=quantity, sources_except=sources_except)

        if isinstance(sources, basestring):
            if sources in groups[quantity]:
                if sources == 'all':
                    srcs = []
                    for src in groups[quantity]['all']:
                        if src in sources_except:
                            continue
                        srcs.append(src)
                else:
                    srcs = groups[quantity][sources]
            else:
                srcs = [sources]
        else:
            srcs = sources

        for source in srcs:
            if source not in data:
                continue

            M = data[source]['M']
            phi = data[source]['phi']
            err = data[source]['err']
            ulim = data[source]['ulim']

            mkw = {'capthick': 1, 'elinewidth': 1, 'alpha': 1.0, 'capsize': 1,
                'mec':default_colors[source],
                'fmt': default_markers[source],
                'color':default_colors[source]}

            if not use_labels:
                label = None
            elif ('label' not in kwargs):
                if 'label' in data[source]:
                    label = data[source]['label']
                else:
                    label = source
            else:
                label = kwargs['label']

            mkw['label'] = label
            mkw.update(kwargs)

            if AUV is not None:
                dc = AUV(z, np.array(M))
            else:
                dc = 0

            # Shift band [optional]
            if quantity in ['lf']:
                if data[source]['wavelength'] != wavelength:
                    #shift = sed_model.
                    print("# WARNING: {0!s} wavelength={1}A, not {2}A!".format(\
                        source, data[source]['wavelength'], wavelength))
            #else:
            if source in ['stefanon2017', 'duncan2014']:
                shift = 0.25
                print("# Shifting stellar masses by 0.25 dex (Chabrier -> Salpeter) for source={}".format(source))
            else:
                shift = 0.

            if log10Mass:
                ax.errorbar(np.log10(M+shift-dc), phi, yerr=err, uplims=ulim, zorder=10, **mkw)
            else:
                ax.errorbar(M+shift-dc, phi, yerr=err, uplims=ulim, zorder=10, **mkw)

        if quantity == 'lf':
            ax.set_xticks(np.arange(-26, 0, 1), minor=True)
            ax.set_xlim(-26.5, -10)
            ax.set_yscale('log')
            ax.set_ylim(1e-7, 1)
            if (not gotax) or force_labels:
                ax.set_xlabel(r'$M_{\mathrm{UV}}$')
                ax.set_ylabel(r'$\phi(M_{\mathrm{UV}}) \ [\mathrm{mag}^{-1} \ \mathrm{cMpc}^{-3}]$')
        elif quantity in ['smf', 'smf_sf', 'smf_q']:

            if log10Mass:
                ax.set_xlim(7, 13)
                if (not gotax) or force_labels:
                    ax.set_xlabel(r'log$_{10}(M_{\ast} / M_{\odot})$')

            else:
                try:
                    ax.set_xscale('log')
                except ValueError:
                    pass
                ax.set_xlim(1e7, 1e13)
                if (not gotax) or force_labels:
                    ax.set_xlabel(r'$M_{\ast} / M_{\odot}$')

            try:
                ax.set_yscale('log')
            except ValueError:
                pass
            ax.set_ylim(1e-7, 1)
            if (not gotax) or force_labels:
                ax.set_xlabel(r'$M_{\ast} / M_{\odot}$')
                ax.set_ylabel(r'$\phi(M_{\ast}) \ [\mathrm{dex}^{-1} \ \mathrm{cMpc}^{-3}]$')

        elif quantity == 'mzr':
            ax.set_xlim(1e8, 1e12)
            ax.set_ylim(7, 9.5)

            if (not gotax) or force_labels:
                ax.set_xlabel(r'$M_{\ast} / M_{\odot}$')
                ax.set_ylabel(r'$12+\log{\mathrm{O/H}}$')
        elif quantity in ['ssfr']:
        	try:
        	    ax.set_xscale('log')
        	    # ax.set_yscale('log')
        	except ValueError:
        	    pass
        	if (not gotax) or force_labels:
        	    ax.set_xlabel(r'$M_{\ast} / M_{\odot}$')
        	    ax.set_ylabel(r'log(SSFR))$ \ [\mathrm{yr}^{-1}]$')


        pl.draw()

        return ax

    #def MultiPlot(self, redshifts, sources='all', round_z=False, ncols=1,
    #    panel_size=(0.75,0.75), fig=1, xmax=-10, ymax=10, legends=None, AUV=None,
    #    quantity='lf', axes=None, sources_except=[],
    #    fig_kwargs={}, show_ylabel=True, **kwargs):
    #    """
    #    Plot the luminosity function at a bunch of different redshifts.
#
    #    Parameters
    #    ----------
    #    z : list
    #        List of redshifts to include.
    #    ncols : int
    #        How many columns in multiplot? Number of rows will be determined
    #        automatically.
    #    legends : bool, str
    #        'individual' means one legend per axis, 'master' means one
    #        (potentially gigantic) legend.
#
    #    """
#
    #    if ncols == 1:
    #        nrows = len(redshifts)
    #    else:
    #        nrows = len(redshifts) // ncols
#
    #    if nrows * ncols != len(redshifts):
    #        nrows += 1
#
    #    dims = (nrows, ncols)
#
    #    # Force redshifts to be in ascending order
    #    if not np.all(np.diff(redshifts)) > 0:
    #        redshifts = np.sort(redshifts)
#
    #    annotate_z = 'left' if quantity == 'lf' else 'right'
#
    #    # Create multiplot
    #    if axes is None:
    #        gotmp = False
    #        fig, axes = pl.subplots(*dims, num=fig, **fig_kwargs)
    #    else:
    #        gotmp = True
#
    #    if not hasattr(self, 'redshifts_in_mp'):
    #        self.redshifts_in_mp = {}
#
    #    if quantity not in self.redshifts_in_mp:
    #        self.redshifts_in_mp[quantity] = []
#
    #    for i, z in enumerate(redshifts):
    #        k = mp.elements.ravel()[i]
    #        ax = mp.grid[k]
#
    #        # Where in the MultiPlot grid are we?
    #        self.redshifts_in_mp[quantity].append(k)
#
    #        self.Plot(z, sources=sources, round_z=round_z, ax=ax, AUV=AUV,
    #            quantity=quantity, sources_except=sources_except, **kwargs)
#
    #        if annotate_z == 'left':
    #            _xannot = 0.05
    #        else:
    #            _xannot = 0.95
#
    #        if gotmp:
    #            continue
#
    #        ax.annotate(r'$z \sim {}$'.format(round(z, 1)), (_xannot, 0.95),
    #            ha=annotate_z, va='top', xycoords='axes fraction')
#
    #    if gotmp:
    #        return mp
#
    #    for i, z in enumerate(redshifts):
    #        k = mp.elements.ravel()[i]
    #        ax = mp.grid[k]
#
    #        if quantity == 'lf':
    #            ax.set_xlim(-24, xmax)
    #            ax.set_ylim(1e-7, ymax)
    #            ax.set_yscale('log', nonposy='clip')
    #            ax.set_ylabel('')
    #            ax.set_xlabel(r'$M_{\mathrm{UV}}$')
    #        else:
    #            ax.set_xscale('log')
    #            ax.set_xlim(1e6, 1e12)
    #            ax.set_ylim(1e-7, ymax)
    #            ax.set_yscale('log', nonposy='clip')
    #            ax.set_xlabel(r'$M_{\ast} / M_{\odot}$')
#
    #    if show_ylabel:
    #        if quantity == 'lf':
    #            mp.global_ylabel(r'$\phi(M_{\mathrm{UV}}) \ [\mathrm{mag}^{-1} \ \mathrm{cMpc}^{-3}]$')
    #        else:
    #            mp.global_ylabel(r'$\phi(M_{\ast}) \ [\mathrm{dex}^{-1} \ \mathrm{cMpc}^{-3}]$')
#
#
    #    pl.show()
#
    #    return mp

    def _selected(self, color1, color2, lbcut, ccut, degen):

        inter, slope = degen

        is_highz = np.logical_and(color1 >= lbcut, color2 <= ccut)

        x = color1#np.arange(lbcut, 3.5, 0.01)
        y = (x - inter) / slope

        is_highz = np.logical_and(color2 <= y, is_highz)

        return is_highz


    def PlotColorColor(self, pop, redshifts=[4,5,6,7], cuts='bouwens2015',
        fig=None, show_false_neg=True): # pragma: no cover
        """
        Make color-color plot including high-z selection criteria.
        """

        Nz = len(redshifts)

        if (fig is None) or (type(fig) is int):
            fig = pl.figure(tight_layout=False,
                figsize=(4*Nz, 4 * (1+show_false_neg)), num=fig)
            fig.subplots_adjust(left=0.15, bottom=0.15, top=0.9, right=0.9)

        gs = gridspec.GridSpec(1+show_false_neg, Nz,
            hspace=0.5, wspace=0.3, figure=fig)

        color_selection = read_lit(cuts).color_selection

        names = read_lit('bouwens2014').filter_names
        cam = ('wfc', 'wfc3')

        phot = {}
        axes = []
        for i, z in enumerate(redshifts):

            ax = fig.add_subplot(gs[0,i])
            ax2 = fig.add_subplot(gs[1,i])
            #ax3 = fig.add_subplot(gs[2,i])
            #axes.append(ax)
            #
            #ax4 = fig2.add_subplot(gs2[0,i])

            ax.annotate(r'$z \sim {}$'.format(z), (0.05, 0.95), ha='left',
                va='top', xycoords='axes fraction')

            cuts = color_selection[z]

            n1A, n1B, n1gt = cuts[0]
            n2A, n2B, n2lt = cuts[1]
            inter, slope = cuts[2]

           # color1 = ph_mags[ph_fil.index(n1A)] - ph_mags[ph_fil.index(n1B)]
           # color2 = ph_mags[ph_fil.index(n2A)] - ph_mags[ph_fil.index(n2B)]

            # Left rectangle: constraint on color2 (y axis)
            ax.fill_betweenx([n2lt, 3.5], -1, 3.5, color='k', alpha=0.2,
                edgecolors='none')
            # Bottom rectangle: constraint on color1 (x axis)
            ax.fill_between([-1,n1gt], -1, n2lt, color='k', alpha=0.2,
                edgecolors='none')

            #y = np.arange(-1, n2lt+0.05, 0.05)
            #x = inter + y * slope
            x = np.arange(n1gt, 3.5, 0.01)
            y = (x - inter) / slope

            ok = y <= n2lt

            ax.fill_between(x[ok==1], y[ok==1], np.ones_like(y[ok==1]) * n2lt,
                color='k', alpha=0.2)

            ax.set_xlabel(r'{}-{}'.format(names[n1A], names[n1B]))
            ax.set_ylabel(r'{}-{}'.format(names[n2A], names[n2B]))

            hist = pop.histories

            dL = pop.cosm.LuminosityDistance(z)
            magcorr = 5. * (np.log10(dL / cm_per_pc) - 1.)

            ph_mags = []
            ph_xph  = []
            ph_dx = []
            ph_fil = []
            for j, _cam in enumerate(cam):

                _filters, xphot, dxphot, ycorr = \
                    pop.synth.Photometry(zobs=z, sfh=hist['SFR'], zarr=hist['z'],
                        hist=hist, dlam=20., cam=_cam, filters=list(names.keys()),
                        extras=pop.extras, rest_wave=None, load=False)

                ph_mags.extend(list(np.array(ycorr) - magcorr))
                ph_xph.extend(xphot)
                ph_dx.extend(list(np.sum(dxphot, axis=1).squeeze()))
                ph_fil.extend(_filters)

            ph_mags = np.array(ph_mags)

            phot[z] = ph_mags

            _color1 = ph_mags[ph_fil.index(n1A)] - ph_mags[ph_fil.index(n1B)]
            _color2 = ph_mags[ph_fil.index(n2A)] - ph_mags[ph_fil.index(n2B)]

            is_highz = self._selected(_color1, _color2, n1gt, n2lt,
                (inter, slope))

            false_neg = (is_highz.size - is_highz.sum()) / float(is_highz.size)
            print('False negatives at z={}: {}'.format(z, false_neg))

            #for _ax in axes:
            ax.scatter(_color1[is_highz==1], _color2[is_highz==1], color='b',
                facecolor='b', edgecolors='none', alpha=0.01)
            ax.scatter(_color1[is_highz==0], _color2[is_highz==0], color='r',
                facecolor='r', edgecolors='none', alpha=0.01)

            ax.set_xlim(-0.5, 3.5)
            ax.set_ylim(-0.5, 2)

            if not show_false_neg:
                continue

            # Plot false negative rate vs. SFR
            sfr = pop.get_field(z, 'SFR')
            Mh = pop.get_field(z, 'Mh')
            sfr_bins = np.arange(-3, 3, 0.2)

            # y is irrelevant here
            if np.any(is_highz==1):
                x1, y1, std1, N1 = bin_samples(np.log10(sfr[is_highz==1]),
                    Mh[is_highz==1],
                    sfr_bins, return_N=True, inclusive=True)
            else:
                N1 = 0

            if np.any(is_highz==0):
                x1, y2, std2, N2 = bin_samples(np.log10(sfr[is_highz==0]),
                    Mh[is_highz==0],
                    sfr_bins, return_N=True, inclusive=True)
            else:
                N2 = 0

            x_all, y_all, std_all, N_all = bin_samples(np.log10(sfr),
                Mh, sfr_bins, return_N=True, inclusive=True)

            tot_by_bin = N1 + N2

            ax2.semilogx(10**x_all, N2 / tot_by_bin.astype('float'), color='k')
            ax2.set_ylim(-0.05, 1.05)
            ax2.set_xlabel(r'$\dot{M}_{\ast} \ [M_{\odot} \ \mathrm{yr}^{-1}]$')

            if i == 0:
                ax2.set_ylabel('false negative rate')

        return fig, gs

    def annotated_legend(self, ax, loc=(0.95, 0.05), sources='all'):
        """
        Annotate sources properly color-coded.
        """
        if sources in groups[quantity]:
            srcs = groups[quantity][sources]
        elif isinstance(sources, basestring):
            srcs = [sources]

        for i, source in enumerate(srcs):
            coord = (loc[0], loc[1] + 0.05 * i)
            ax.annotate(source, coord, fontsize=14,
                color=default_colors[source], ha='right', va='bottom',
                xycoords='axes fraction')

        pl.draw()

        return ax

    def PlotSummary(self, pop, axes=None, fig=1, use_best=True, method='mode',
        fresh=False, redshifts=None, include_colors=True, **kwargs): # pragma: no cover
        """
        Make a huge plot.
        """

        if axes is None:
            gotax = False
            axes = self._MegaPlotSetup(fig)
        else:
            gotax = True

        if not gotax:
            self._MegaPlotCalData(axes, redshifts=redshifts)
            self._MegaPlotPredData(axes, redshifts=redshifts)
            self._MegaPlotGuideEye(axes, redshifts=redshifts)

        if pop is None:
            pass
        elif isinstance(pop, GalaxyEnsemble):
            self._pop = pop
            self._MegaPlotPop(axes, pop, redshifts=redshifts)
        elif hasattr(pop, 'chain'):
            if fresh:
                bkw = pop.base_kwargs.copy()
                bkw.update(pop.max_likelihood_parameters(method=method))
                pop = GP(**bkw)
                self._pop = pop
                self._MegaPlotPop(axes, pop, redshifts=redshifts,
                    include_colors=include_colors)
            else:
                self._MegaPlotChain(axes, pop, use_best=use_best, **kwargs)
        else:
            raise NotImplemented("Unrecognized object pop={}".format(pop))

        self._MegaPlotCleanup(pop, axes)

        return axes

    def _MegaPlotPop(self, kw, pop, redshifts=None, include_colors=True,
        **kwargs):


        ax_sfe = kw['ax_sfe']
        ax_fco = kw['ax_fco']
        ax_rdu = kw['ax_rdu']
        ax_phi = kw['ax_phi']
        ax_bet = kw['ax_bet']

        ax_smf    = kw['ax_smf']
        #ax_smhm   = kw['ax_smhm']
        ax_MsMUV  = kw['ax_MsMUV']
        ax_AUV    = kw['ax_AUV']
        ax_sfrd   = kw['ax_sfrd']
        #ax_lae_z  = kw['ax_lae_z']
        #ax_lae_m  = kw['ax_lae_m']
        ax_sfms   = kw['ax_sfms']

        _mst = np.arange(6, 14, 0.2)
        _mh = np.logspace(6, 13, 100)
        _mags = np.arange(-25, -10, pop.pf['pop_mag_bin'])

        if redshifts is None:
            redshifts = [4, 6, 8, 10]

        colors = ['k', 'b', 'c', 'm', 'g', 'y', 'r']

        dc1 = DustCorrection(dustcorr_method='meurer1999',
            dustcorr_beta='bouwens2014')

        xa_b = []
        xa_f = []
        for j, z in enumerate(redshifts):

            # UVLF
            _mags_, phi = pop.LuminosityFunction(z, _mags)
            ax_phi.semilogy(_mags, phi, color=colors[j], drawstyle='steps-mid')

            # Binned version
            if z <= 7:
                Mbins = np.arange(-25, -10, 1.0)
                if pop.pf['pop_dust_yield'] is not None:
                    _beta = pop.Beta(z, Mwave=1600., return_binned=True,
                        Mbins=Mbins, presets='hst', rest_wave=None, dlam=20.)
                else:
                    _beta = np.zeros_like(Mbins)

                if np.any(_beta != -99999):
                    ax_bet.plot(Mbins, _beta, color=colors[j])

            Mh = pop.get_field(z, 'Mh')
            Ms = pop.get_field(z, 'Ms')
            nh = pop.get_field(z, 'nh')
            SFR = pop.get_field(z, 'SFR')

            SFE = pop.guide.SFE(z=z, Mh=_mh)

            ax_sfe.loglog(_mh, SFE, color=colors[j], alpha=0.8,
                label=r'$z={}$'.format(z))

            if (pop.pf['pop_scatter_mar'] > 0) or (pop.pf['pop_histories'] is not None):
                _bins = np.arange(7, 12.1, 0.1)
                x, y, std, N = bin_samples(np.log10(Ms), np.log10(SFR), _bins,
                    weights=nh)
                ax_sfms.loglog(10**x, 10**y, color=colors[j])
            else:
                ax_sfms.loglog(Ms, SFR, color=colors[j])

            # SMF
            phi = pop.StellarMassFunction(z, _mst)
            ax_smf.loglog(10**_mst, phi, color=colors[j], drawstyle='steps-mid')

            # SMHM
            _Mh = 10**np.arange(8, 12.5, 0.1)
            fstar = pop.SMHM(z, _Mh, return_mean_only=True)
            #ax_smhm.loglog(_Mh, 10**fstar, color=colors[j])

            if not include_colors:
                continue

            filt, mags1500 = pop.Magnitude(z, wave=1500.)


            #mags = pop.Magnitude(z, wave=1600.)
            #if pop.pf['pop_dust_yield'] is not None:
            #    beta = pop.Beta(z, Mwave=1600., return_binned=False)
            #else:
            #    beta = np.zeros_like(mags)

            # M1500-Mstell
            _x, _y, _z, _N = bin_samples(mags1500, np.log10(Ms), Mbins,
                weights=nh)
            ax_MsMUV.plot(_x, _y, color=colors[j])

            # Beta just to get 'mags'
            if pop.pf['pop_dust_yield'] in [0, None]:
                xa_f.append(0)
                xa_b.append(0)

                if pop.pf['dustcorr_method'] is not None:
                    print("dustcorr_method={}".format(pop.pf['dustcorr_method']))
                    ax_bet.plot(Mbins, dc1.Beta(z, Mbins), color=colors[j])

                continue


            Rdust = pop.guide.dust_scale(z=z, Mh=Mh)
            ydust = pop.guide.dust_yield(z=z, Mh=Mh)

            if pop.pf['pop_fduty'] is not None:
                fduty = pop.guide.fduty(z=z, Mh=Mh)
            else:
                fduty = np.zeros_like(Mh)

            if pop.pf['pop_dust_growth'] not in [0, None]:
                fgrowth = pop.guide.dust_growth(z=z, Mh=Mh)
            else:
                fgrowth = np.zeros_like(Mh)

            #any_fcov = np.any(np.diff(fcov, axis=1) != 0)
            #any_fduty = np.any(np.diff(fduty, axis=1) != 0)

            if type(pop.pf['pop_dust_yield']) is str:
                ax_fco.semilogx(Mh, ydust, color=colors[j], ls='--')
                ax_fco.set_ylabel(r'$y_{\mathrm{dust}}$')
            if type(pop.pf['pop_fduty']) is str:
                ax_fco.semilogx(Mh, fduty, color=colors[j])
                ax_fco.set_ylabel(r'$f_{\mathrm{duty}}$')
            if type(pop.pf['pop_dust_growth']) is str:
                ax_fco.semilogx(Mh, fgrowth, color=colors[j])
                ax_fco.set_ylabel(r'$f_{\mathrm{growth}}$')

            ax_rdu.loglog(Mh, Rdust, color=colors[j])

            Mbins = np.arange(-25, -10, 1.)
            AUV = pop.AUV(z, Mwave=1600., return_binned=True,
                magbins=Mbins)

            ax_AUV.plot(Mbins, AUV, color=colors[j])

            # LAE stuff
            #_x, _y, _z, _N = bin_samples(mags, fcov, Mbins)
            #ax_lae_m.plot(_x, 1. - _y, color=colors[j])
            #
            #faint  = np.logical_and(Mbins >= -20.25, Mbins < -18.)
            #bright = Mbins < -20.25
            #
            #xa_f.append(1. - np.mean(_y[faint==1]))
            #xa_b.append(1. - np.mean(_y[bright==1]))

        #ax_lae_z.plot(redshifts, xa_b, color='k', alpha=1.0, ls='-')
        #ax_lae_z.plot(redshifts, xa_f, color='k', alpha=1.0, ls='--')

        zarr = np.arange(4, 25, 0.1)
        sfrd = np.array([pop.SFRD(zarr[i]) for i in range(zarr.size)])
        ax_sfrd.semilogy(zarr, sfrd * rhodot_cgs, color='k')

    def _MegaPlotChain(self, kw, anl, **kwargs):
        """
        Plot many samples
        """

        ax_sfe = kw['ax_sfe']
        ax_fco = kw['ax_fco']
        ax_rdu = kw['ax_rdu']
        ax_phi = kw['ax_phi']
        ax_bet = kw['ax_bet']


        ax_smf    = kw['ax_smf']
        #ax_smhm   = kw['ax_smhm']
        ax_MsMUV  = kw['ax_MsMUV']
        ax_AUV    = kw['ax_AUV']
        ax_sfrd   = kw['ax_sfrd']
        #ax_lae_z  = kw['ax_lae_z']
        #ax_lae_m  = kw['ax_lae_m']
        ax_sfms   = kw['ax_sfms']

        _mst  = np.arange(6, 12.25, 0.25)
        _mags = np.arange(-25, -10, anl.base_kwargs['pop_mag_bin'])

        redshifts = [4, 6, 8, 10]
        colors = ['k', 'b', 'c', 'm', 'g', 'y', 'r']

        dc1 = DustCorrection(dustcorr_method='meurer1999',
            dustcorr_beta='bouwens2014')

        xa_b = []
        xa_f = []
        for j, z in enumerate(redshifts):

            # UVLF
            anl.ReconstructedFunction('galaxy_lf', ivar=[z, None], ax=ax_phi,
                color=colors[j], **kwargs)

            anl.ReconstructedFunction('fstar', ivar=[z, None], ax=ax_sfe,
                color=colors[j], **kwargs)

            if 'galaxy_smf' in anl.all_blob_names:
                anl.ReconstructedFunction('galaxy_smf', ivar=[z, None], ax=ax_smf,
                    color=colors[j], is_logx=True, **kwargs)

            #if 'MUV_gm' in anl.all_blob_namess:
            #    _z, _MUV = anl.get_ivars('MUV_gm')
            #    k = np.argmin(np.abs(z - _z))
            #    new_x = anl.ExtractData('MUV_gm')['MUV_gm'][:,k,:]
            #    print("New magnitudes!!!!")
            #else:
            new_x = None

            anl.ReconstructedFunction('sfrd', ivar=None, ax=ax_sfrd,
                color=colors[j], multiplier=rhodot_cgs, **kwargs)

            if 'pop_dust_yield' not in anl.base_kwargs:
                continue

            dtmr = anl.base_kwargs['pop_dust_yield']
            if (dtmr is None) or (dtmr == 0):
                continue

            anl.ReconstructedFunction('beta_hst', ivar=[z, None], ax=ax_bet,
                color=colors[j], new_x=new_x, **kwargs)

            anl.ReconstructedFunction('AUV', ivar=[z, None], ax=ax_AUV,
                color=colors[j], **kwargs)

            anl.ReconstructedFunction('dust_scale', ivar=[z, None], ax=ax_rdu,
                color=colors[j], **kwargs)

            if 'fduty' in anl.all_blob_names:
                anl.ReconstructedFunction('fduty', ivar=[z, None], ax=ax_fco,
                    color=colors[j], **kwargs)

            if 'dust_yield' in anl.all_blob_names:
                anl.ReconstructedFunction('dust_yield', ivar=[z, None], ax=ax_fco,
                    color=colors[j], ls='--', **kwargs)

            if 'fgrowth' in anl.all_blob_names:
                anl.ReconstructedFunction('fgrowth', ivar=[z, None], ax=ax_fco,
                    color=colors[j], **kwargs)
                ax_fco.set_yscale('log')
                ax_fco.set_ylim(1e9, 1e13)


    def _MegaPlotLimitsAndTicks(self, anl, kw):
        ax_sfe = kw['ax_sfe']
        ax_fco = kw['ax_fco']
        ax_rdu = kw['ax_rdu']
        ax_phi = kw['ax_phi']
        ax_bet = kw['ax_bet']

        ax_smf    = kw['ax_smf']
        #ax_smhm   = kw['ax_smhm']
        ax_MsMUV  = kw['ax_MsMUV']
        ax_AUV    = kw['ax_AUV']
        ax_sfrd   = kw['ax_sfrd']
        #ax_lae_z  = kw['ax_lae_z']
        #ax_lae_m  = kw['ax_lae_m']
        ax_sfms   = kw['ax_sfms']



        ax_sfe.set_xlim(1e8, 1e13)
        ax_sfe.set_ylim(1e-3, 1.5)
        ax_fco.set_xscale('log')
        ax_fco.set_xlim(1e8, 1e13)
        ax_fco.set_yscale('linear')

        if anl is not None:
            if 'pop_dust_growth' in anl.pf:
                ax_fco.set_yscale('log')
                ax_fco.set_ylim(1e9, 1e13)
            else:
                ax_fco.set_ylim(0, 1.05)

            if ('dust_scale' in anl.all_blob_names) and ('fduty' in anl.all_blob_names):
                ax_fco.set_ylabel(r'$f_{\mathrm{duty}}$')
                ax_fco2 = ax_fco.twinx()
                ax_fco2.set_ylabel(r'$f_{\mathrm{dtmr}}$')
                ax_fco2.set_ylim(0, 1.05)
        else:
            ax_fco.set_ylim(0, 1.05)

        ax_rdu.set_xlim(1e8, 1e13)
        ax_rdu.set_ylim(1e-2, 100)

        ax_smf.set_xscale('log')
        ax_smf.set_xlim(1e7, 1e12)
        ax_smf.set_ylim(1e-7, 2e-1)
        #ax_smhm.set_xscale('log')
        #ax_smhm.set_yscale('log')
        #ax_smhm.set_ylim(-4, 1.)
        #ax_smhm.set_yscale('log', nonposy='clip')
        #ax_smhm.set_xlim(1e9, 1e12)
        #ax_smhm.set_ylim(5e-4, 1.5e-1)
        ax_bet.set_xlim(-25, -12)
        ax_bet.set_ylim(-3, -1)
        ax_phi.set_xlim(-25, -12)
        ax_phi.set_ylim(1e-7, 2e-1)

        ax_MsMUV.set_yscale('linear')
        ax_MsMUV.set_ylim(7, 12)
        ax_MsMUV.set_xlim(-25, -12)

        ax_AUV.set_xlim(-25, -12)
        ax_AUV.set_ylim(0, 3.5)

        ax_sfms.set_xlim(1e7, 1e12)
        ax_sfms.set_ylim(1e-2, 2e3)

        #ax_lae_m.set_xlim(-25, -12)
        #ax_lae_z.set_xlim(3., 7.2)
        #ax_lae_m.set_ylim(-0.05, 1.05)
        #ax_lae_z.set_ylim(-0.05, 1.05)

        ax_sfrd.set_yscale('log')
        ax_sfrd.set_xlim(4, 20)
        ax_sfrd.set_ylim(1e-4, 1e-1)

        # Set ticks for all MUV scales
        for ax in [ax_bet, ax_phi, ax_MsMUV, ax_AUV]:
            ax.set_xticks(np.arange(-24, -12, 1), minor=True)

        for ax in [ax_MsMUV, ax_AUV]:
            ax.set_xlim(-25, -15)

        return kw

    def _MegaPlotSetup(self, fig):

        fig = pl.figure(tight_layout=False, figsize=(22, 7), num=fig)
        #gs = gridspec.GridSpec(3, 10, hspace=0.3, wspace=1.0)
        gs = gridspec.GridSpec(3, 14, hspace=0.3, wspace=5.0)

        # Inputs
        ax_sfe = fig.add_subplot(gs[0,0:3])
        ax_rdu = fig.add_subplot(gs[1,0:3])
        ax_fco = fig.add_subplot(gs[2,0:3])

        # Rest UV stuff / calibration
        ax_phi = fig.add_subplot(gs[0:2,3:7])
        ax_bet = fig.add_subplot(gs[2,3:7])

        # Predictions
        ax_smf = fig.add_subplot(gs[0:2,7:11])
        ax_sfms = fig.add_subplot(gs[2,7:11])

        #ax_smhm = fig.add_subplot(gs[2,12:])
        ax_AUV = fig.add_subplot(gs[0,11:14])
        ax_MsMUV = fig.add_subplot(gs[1,11:14])
        ax_sfrd = fig.add_subplot(gs[2,11:14])

        # Cal



        # Placeholder
        #ax_tau = fig.add_subplot(gs[0:1,9])

        kw = \
        {
         'ax_sfe': ax_sfe,
         'ax_fco': ax_fco,
         'ax_rdu': ax_rdu,
         'ax_phi': ax_phi,
         'ax_bet': ax_bet,
         'ax_smf': ax_smf,
         'ax_MsMUV': ax_MsMUV,
         'ax_AUV': ax_AUV,
         'ax_sfrd': ax_sfrd,
         'ax_sfms': ax_sfms,
        }

        return kw

    def _MegaPlotCalData(self, kw, redshifts=None):

        ax_sfe = kw['ax_sfe']
        ax_fco = kw['ax_fco']
        ax_rdu = kw['ax_rdu']
        ax_phi = kw['ax_phi']
        ax_bet = kw['ax_bet']


        ax_smf    = kw['ax_smf']
        #ax_smhm   = kw['ax_smhm']
        ax_MsMUV  = kw['ax_MsMUV']
        ax_AUV    = kw['ax_AUV']
        ax_sfrd   = kw['ax_sfrd']
        #ax_lae_z  = kw['ax_lae_z']
        #ax_lae_m  = kw['ax_lae_m']
        ax_sfms   = kw['ax_sfms']


        l11 = read_lit('lee2011')
        b14 = read_lit('bouwens2014')

        # Vanilla dust model
        dc1 = DustCorrection(dustcorr_method='meurer1999',
            dustcorr_beta='bouwens2014')
        #devol = ares.util.ParameterBundle('dust:evolving')
        #dc2 = ares.phenom.DustCorrection(**devol)
        #dc3 = DustCorrection(dustcorr_method='pettini1998',
        #    dustcorr_beta='bouwens2014')


        # Redshifts and color scheme
        if redshifts is None:
            redshifts = [4, 6, 8, 10]

        colors = ['k', 'b', 'c', 'm', 'g', 'y', 'r']
        mkw = {'capthick': 1, 'elinewidth': 1, 'alpha': 1.0, 'capsize': 1}

        # UVLF and Beta
        for j, z in enumerate(redshifts):
            self.PlotLF(z, ax=ax_phi, sources=['bouwens2015'],
                round_z=0.21, color=colors[j], mec=colors[j], mfc=colors[j], fmt='o',
                label='Bouwens+ 2015' if j == 0 else None, **mkw)
            self.PlotLF(z, ax=ax_phi, sources=['oesch2018'],
                round_z=0.21, color=colors[j], mec=colors[j], mfc=colors[j], fmt='d',
                label='Oesch+ 2018' if j == 0 else None, **mkw)
            self.PlotLF(z, ax=ax_phi, sources=['finkelstein2015'],
                round_z=0.21, color=colors[j], mec=colors[j], mfc='none', mew=1, fmt='s',
                label='Finkelstein+ 2015' if j == 0 else None, **mkw)
            self.PlotSMF(z, ax=ax_smf, sources=['song2016'],
                round_z=0.11, color=colors[j], mec=colors[j], mfc=colors[j], mew=1, fmt='o',
                label='Song+ 2016' if j == 0 else None, **mkw)
            self.PlotSMF(z, ax=ax_smf, sources=['stefanon2017'], mew=1, fmt='s',
                round_z=0.11, color=colors[j], mec=colors[j], mfc='none',
                label='Stefanon+ 2017' if j == 0 else None, **mkw)
            self.PlotSMF(z, ax=ax_smf, sources=['duncan2014'],
                round_z=0.11, color=colors[j], mec=colors[j], mfc=colors[j], mew=1, fmt='o',
                label='Duncan+ 2014' if j == 0 else None, **mkw)

            if z in b14.data['beta']:

                err = b14.data['beta'][z]['err'] + b14.data['beta'][z]['sys']
                ax_bet.errorbar(b14.data['beta'][z]['M'], b14.data['beta'][z]['beta'], err,
                    fmt='o', color=colors[j], label=r'Bouwens+ 2014' if j == 0 else None,
                    **mkw)

            if z in l11.data['beta']:
                ax_bet.errorbar(l11.data['beta'][z]['M'], l11.data['beta'][z]['beta'],
                    l11.data['beta'][z]['err'],
                    fmt='*', color=colors[j], label=r'Lee+ 2011' if j == 0 else None,
                    **mkw)

            # Plot vanilla dust correction
            ax_AUV.plot(np.arange(-25, -15, 0.1),
                dc1.AUV(z, np.arange(-25, -15, 0.1)),
                color=colors[j], ls=':',
                label=r'M99+B14 IRX-$\beta + M_{\mathrm{UV}}-\beta$' if j == 0 else None)


            #ax_AUV.plot(np.arange(-25, -14, 2.), dc2.AUV(z, np.arange(-25, -14, 2.)),
            #    color=colors[j], ls='--',
            #    label=r'evolving IRX-$\beta + M_{\mathrm{UV}}-\beta$' if j == 0 else None)
            #ax_AUV.plot(np.arange(-25, -14, 2.), dc3.AUV(z, np.arange(-25, -14, 2.)),
            #    color=colors[j], ls='-.',
            #    label=r'P98+B14 IRX-$\beta + M_{\mathrm{UV}}-\beta$' if j == 0 else None)

    def _MegaPlotGuideEye(self, kw, redshifts=None):
        ax_sfe = kw['ax_sfe']
        ax_fco = kw['ax_fco']
        ax_rdu = kw['ax_rdu']
        ax_phi = kw['ax_phi']
        ax_bet = kw['ax_bet']


        ax_smf    = kw['ax_smf']
        #ax_smhm   = kw['ax_smhm']
        ax_MsMUV  = kw['ax_MsMUV']
        ax_AUV    = kw['ax_AUV']
        ax_sfrd   = kw['ax_sfrd']
        #ax_lae_z  = kw['ax_lae_z']
        #ax_lae_m  = kw['ax_lae_m']
        ax_sfms   = kw['ax_sfms']

        ax_rdu.annotate(r'$R_h \propto M_h^{1/3} (1+z)^{-1}$', (1.5e8, 30))

        if redshifts is None:
            redshifts = [4, 6, 8, 10]

        colors = ['k', 'b', 'c', 'm', 'g', 'y', 'r']

        # Show different Mh slopes
        mh = np.logspace(8, 9, 50)

        # How Rdust would scale if it were proportional with halo size
        for j, z in enumerate(redshifts):
            ax_rdu.loglog(mh, 5. * (mh / 1e8)**0.333 * (1. + 4.) / (1. + z), color=colors[j],
                lw=1, ls='-', alpha=0.5)


        func = lambda z, A: 5e-2 * (mh / 1e8)**A #* (1. + 4.) / (1. + zz)**B
        ax_sfe.loglog(mh, func(4., 1./3.),
            color='k', lw=1, ls='-', alpha=0.5)
        ax_sfe.loglog(mh, func(4., 2./3.),
            color='k', lw=1, ls='-', alpha=0.5)
        ax_sfe.loglog(mh, func(4., 3./3.),
            color='k', lw=1, ls='-', alpha=0.5)
        ax_sfe.annotate(r'$1/3$', (mh[-1]*1.1, func(4., 1./3.)[-1]), ha='left')
        ax_sfe.annotate(r'$2/3$', (mh[-1]*1.1, func(4., 2./3.)[-1]), ha='left')
        ax_sfe.annotate(r'$1$',   (mh[-1]*1.1, func(4., 3./3.)[-1]), ha='left')

        # Show different z-dep
        ax_sfe.scatter(np.ones_like(redshifts) * 1e10, 4e-3 * ((1. + np.array(redshifts)) / 9.),
            color=colors, facecolors='none', marker='s', s=5)
        ax_sfe.scatter(np.ones_like(redshifts) * 1e11, 4e-3 * np.sqrt(((1. + np.array(redshifts)) / 9.)),
            color=colors, facecolors='none', marker='s', s=5)
        ax_sfe.annotate(r'$(1+z)$', (1e10, 5e-3), ha='center', va='bottom',
            rotation=0, fontsize=8)
        ax_sfe.annotate(r'$\sqrt{1+z}$', (1e11, 5e-3), ha='center', va='bottom',
            rotation=0, fontsize=8)


        ax_phi.legend(loc='lower right', fontsize=8)
        ax_smf.legend(loc='lower left', fontsize=8)
        ax_bet.legend(loc='upper right', fontsize=8)
        ax_AUV.legend(loc='upper right', fontsize=8)


        # Show different z-dep
        ax_sfms.scatter(np.ones_like(redshifts) * 2e9, 1e-1 * ((1. + np.array(redshifts)) / 9.)**1.5,
            color=colors, facecolors='none', marker='s', s=5)
        ax_sfms.annotate(r'$(1+z)^{3/2}$', (2e9, 1.5e-1), ha='center', va='bottom',
            rotation=0, fontsize=8)
        ax_sfms.scatter(np.ones_like(redshifts) * 2e10, 1e-1 * ((1. + np.array(redshifts)) / 9.)**2.5,
            color=colors, facecolors='none', marker='s', s=5)
        ax_sfms.annotate(r'$(1+z)^{5/2}$', (2e10, 1.5e-1), ha='center', va='bottom',
            rotation=0, fontsize=8)

        mh = np.logspace(7., 8, 50)
        ax_sfms.loglog(mh, 200 * func(4., 3./3.),
            color=colors[0], lw=1, ls='-', alpha=0.5)
        ax_sfms.annotate(r'$1$',   (mh[-1]*1.1, 200 * func(4., 3./3.)[-1]), ha='left')

    def _MegaPlotPredData(self, kw, redshifts=None):

        ax_sfe = kw['ax_sfe']
        ax_fco = kw['ax_fco']
        ax_rdu = kw['ax_rdu']
        ax_phi = kw['ax_phi']
        ax_bet = kw['ax_bet']

        ax_smf    = kw['ax_smf']
        #ax_smhm   = kw['ax_smhm']
        ax_MsMUV  = kw['ax_MsMUV']
        ax_AUV    = kw['ax_AUV']
        ax_sfrd   = kw['ax_sfrd']
        #ax_lae_z  = kw['ax_lae_z']
        #ax_lae_m  = kw['ax_lae_m']
        ax_sfms   = kw['ax_sfms']

        mkw = {'capthick': 1, 'elinewidth': 1, 'alpha': 1.0, 'capsize': 1}

        if redshifts is None:
            redshifts = [4, 6, 8, 10]

        colors = ['k', 'b', 'c', 'm', 'g', 'y', 'r']

        xarr = np.arange(-22, -18, 0.5)
        yarr = [0.1, 0.08, 0.08, 0.1, 0.18, 0.3, 0.47, 0.6]
        yerr = [0.1, 0.05, 0.03, 0.05, 0.05, 0.1, 0.15, 0.2]
        #ax_lae_m.errorbar(xarr, yarr, yerr=yerr, color='k',
        #    label='Stark+ 2010 (3 < z < 6.2)', fmt='o', **mkw)

        zlist = [4., 5, 6.1]
        x25_b = [0.13, 0.25, 0.2]
        x25_f = [0.35, 0.48, 0.55]
        err_b = [0.05, 0.05, 0.08]
        err_f = [0.05, 0.1, 0.15]

        #_colors = 'k', 'g', 'b'
        #for j, z in enumerate(zlist):
        #    ax_lae_z.errorbar(zlist[j], x25_b[j], yerr=err_b[j],
        #        color=_colors[j], ms=5,
        #        label=r'Stark+ 2011' if j == 0 else None,
        #        fmt='s', mfc='none', **mkw)
        #    ax_lae_z.errorbar(zlist[j], x25_f[j], yerr=err_f[j],
        #        color=_colors[j], ms=5,
        #        fmt='o', mfc='none', **mkw)


        ## De Barros et al. (2017)
        #ax_lae_z.errorbar(5.9, 0.1, 0.05, color='b', fmt='*', mfc='none', ms=5,
        #    label=r'deBarros+ 2017', **mkw)
        #ax_lae_z.errorbar(5.9, 0.38, 0.12, color='b', fmt='*', mfc='none', ms=5,
        #    **mkw)
        #
        #ax_lae_z.legend(loc='upper left', frameon=True, fontsize=6)
        #ax_lae_m.legend(loc='upper left', frameon=True, fontsize=6)

        # Salmon et al. 2015
        data = \
        {
         4: {'MUV': np.arange(-21.5, -18, 0.5),
             'Ms': [9.61, 9.5, 9.21, 9.13, 8.96, 8.81, 8.75],
             'err': [0.39, 0.57, 0.47, 0.51, 0.56, 0.53, 0.57]},
         5: {},
         6: {'MUV': np.arange(-21.5, -18.5, 0.5),
             'Ms': [9.34, 9.23, 9.21, 9.14, 8.90, 8.77],
             'err': [0.44, 0.38, 0.41, 0.38, 0.38, 0.47]},
        }

        for j, z in enumerate(redshifts):
            if z not in data:
                continue

            if ('MUV' not in data[z]) or ('Ms' not in data[z]):
                continue

            ax_MsMUV.errorbar(data[z]['MUV'], data[z]['Ms'], yerr=data[z]['err'],
                color=colors[j], label='Salmon+ 2015' if j==0 else None,
                fmt='o', mfc='none', **mkw)

        ax_MsMUV.legend(loc='upper right', fontsize=8)

    def _MegaPlotCleanup(self, anl, kw):


        ax_sfe = kw['ax_sfe']
        ax_fco = kw['ax_fco']
        ax_rdu = kw['ax_rdu']
        ax_phi = kw['ax_phi']
        ax_bet = kw['ax_bet']


        ax_smf    = kw['ax_smf']
        #ax_smhm   = kw['ax_smhm']
        ax_MsMUV  = kw['ax_MsMUV']
        ax_AUV    = kw['ax_AUV']
        ax_sfrd   = kw['ax_sfrd']
        #ax_lae_z  = kw['ax_lae_z']
        #ax_lae_m  = kw['ax_lae_m']
        ax_sfms   = kw['ax_sfms']

        ax_sfe.set_title('Model Inputs', fontsize=18)
        ax_sfe.set_ylabel(r'$f_{\ast} \equiv \dot{M}_{\ast} / f_b \dot{M}_h$')

        ax_fco.set_xlabel(r'$M_h / M_{\odot}$')

        ax_rdu.set_ylabel(r'$R_{\mathrm{dust}} \ [\mathrm{kpc}]$')

        ax_AUV.set_title('Predictions', fontsize=18)
        ax_smf.set_title('Predictions', fontsize=18)

        ax_smf.set_ylabel(labels['galaxy_smf'])
        #ax_smhm.set_xlabel(r'$M_h / M_{\odot}$')
        #ax_smhm.set_ylabel(r'$M_{\ast} / M_h$')
        ax_phi.set_ylabel(labels['galaxy_lf'])
        ax_phi.set_yscale('log')
        ax_bet.set_ylabel(r'$\beta$')

        ax_MsMUV.set_ylabel(r'$\log_{10} M_{\ast} / M_{\odot}$')
        ax_MsMUV.set_xlabel(r'$M_{1500}$')

        ax_AUV.set_xlabel(r'$M_{\mathrm{UV}}$')
        ax_AUV.set_ylabel(r'$A_{\mathrm{UV}}$')

        ax_smf.set_yscale('log')

        ax_sfms.set_xlabel(r'$M_{\ast} / M_{\odot}$')
        ax_sfms.set_ylabel(r'$\dot{M}_{\ast} \ [M_{\odot} \ \mathrm{yr}^{-1}]$')

        ax_sfrd.set_xlabel(r'$z$')
        ax_sfrd.set_ylabel(labels['sfrd'])
        ax_sfrd.set_ylim(1e-4, 1e-1)

        #ax_lae_z.set_xlabel(r'$z$')
        #ax_lae_z.set_ylabel(r'$X_{\mathrm{LAE}}, 1 - f_{\mathrm{cov}}$')
        #ax_lae_m.set_xlabel(r'$M_{\mathrm{UV}}$')
        #ax_lae_m.set_ylabel(r'$X_{\mathrm{LAE}}, 1 - f_{\mathrm{cov}}$')


        ##
        # CALIBRATION DATA
        ##
        ax_phi.set_title('Calibration Data', fontsize=18)
        ax_bet.set_xlabel(r'$M_{\mathrm{UV}}$')
        ax_phi.set_ylabel(labels['lf'])
        ax_bet.set_ylabel(r'$\beta$')

        ax_phi.legend(loc='lower right', fontsize=8)
        ax_smf.legend(loc='lower left', fontsize=8)
        ax_bet.legend(loc='lower left', fontsize=8)
        ax_AUV.legend(loc='upper right', fontsize=8)
        #ax_lae_z.legend(loc='upper left', frameon=True, fontsize=6)
        #ax_lae_m.legend(loc='upper left', frameon=True, fontsize=6)
        ax_MsMUV.legend(loc='upper right', fontsize=8)


        self._MegaPlotLimitsAndTicks(anl, kw)
