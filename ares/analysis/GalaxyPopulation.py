"""

GalaxyPopulation.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Thu Jan 28 12:38:11 PST 2016

Description: 

"""

import numpy as np
from ..util import labels
from ..util import read_lit
import matplotlib.pyplot as pl
from .ModelSet import ModelSet
from ..util.Survey import Survey
from ..phenom import DustCorrection
from matplotlib.patches import Patch
from ..util.ReadData import read_lit
from ..util.Aesthetics import labels
from scipy.optimize import curve_fit
import matplotlib.gridspec as gridspec
from ..physics.Constants import rhodot_cgs
from ..util.SpectralSynthesis import what_filters
from .MultiPlot import MultiPanel, add_master_legend
from ..util.Stats import symmetrize_errors, bin_samples
from ..populations.GalaxyPopulation import GalaxyPopulation as GP
from ..populations.GalaxyEnsemble import GalaxyEnsemble

try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

datasets_lf = ('oesch2013', 'oesch2014', 'bouwens2015', 'atek2015', 
    'parsa2016', 'finkelstein2015', 'vanderburg2010', 'alavi2016', 
    'reddy2009', 'weisz2014', 'bouwens2017', 'oesch2018')
datasets_smf = ('song2016', 'tomczak2014', 'stefanon2017')
datasets_mzr = ('sanders2015',)

groups_lf = \
{
 'dropouts': ('oesch2013', 'oesch2014', 'bouwens2015', 'parsa2016', 
    'finkelstein2015', 'vanderburg2010', 'reddy2009', 'oesch2018'),
 'lensing': ('alavi2016', 'atek2015', 'bouwens2017'),
 'local': ('weisz2014,'),
 'all': datasets_lf,
}

groups_smf = {'all': datasets_smf}
groups = {'lf': groups_lf, 'smf': groups_smf, 'smf_sf': groups_smf, 
    'smf_tot': groups_smf, 
    'mzr': {'all': datasets_mzr}}

colors_cyc = ['m', 'c', 'r', 'y', 'g', 'b'] * 3
markers = ['o'] * 6 + ['s'] * 6    
    
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

_ulim_tick = 0.5

class GalaxyPopulation(object):
    def __init__(self):
        pass

    def compile_data(self, redshift, sources='all', round_z=False,
        quantity='lf', sources_except=[], just_above=True):
        """
        Create a master dictionary containing the MUV points, phi points,
        and (possibly asymmetric) errorbars for all (or some) data available.
        
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
            
            if quantity in ['lf']:
                data[source]['wavelength'] = src.wavelength
                        
            
                        
            M = src.data[quantity][z]['M']            
            if hasattr(M, 'data'):
                data[source]['M'] = M.data
            else:
                data[source]['M'] = np.array(M)
            
            if src.units[quantity] == 'log10':
                err_lo = []; err_hi = []; uplims = []
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
                    
                data[source]['err'] = (err_lo, err_hi) 
                if hasattr(src.data[quantity][z]['phi'], 'data'):       
                    data[source]['phi'] = 10**src.data[quantity][z]['phi'].data
                else:
                    data[source]['phi'] = 10**np.array(src.data[quantity][z]['phi'])
                data[source]['ulim'] = uplims
            else:                
                
                if hasattr(src.data[quantity][z]['phi'], 'data'):
                    data[source]['phi'] = src.data[quantity][z]['phi'].data
                else:
                    data[source]['phi'] = np.array(src.data[quantity][z]['phi'])
                
                err_lo = []; err_hi = []; uplims = []
                for i, err in enumerate(src.data[quantity][z]['err']):
                    
                    if type(err) in [list, tuple, np.ndarray]:
                        err_hi.append(err[0])
                        err_lo.append(err[1])
                        uplims.append(False)
                    elif err is None:
                        err_lo.append(0)
                        err_hi.append(0)
                        uplims.append(False)
                    else:    
                        if (err < 0):
                            err_hi.append(0.0)
                            err_lo.append(_ulim_tick * data[source]['phi'][i])
                        else:
                            err_hi.append(err)
                            err_lo.append(err)
                            
                        uplims.append(err < 0)    
                
                data[source]['ulim'] = uplims
                data[source]['err'] = (err_lo, err_hi)
                
        return data
                
    def PlotLF(self, z, ax=None, fig=1, sources='all', round_z=False, 
            AUV=None, wavelength=1600., sed_model=None, force_labels=False, **kwargs):
                
        return self.Plot(z=z, ax=ax, fig=fig, sources=sources, round_z=round_z,
            AUV=AUV, wavelength=1600, sed_model=None, quantity='lf', 
            force_labels=force_labels, **kwargs)  
        
    def PlotSMF(self, z, ax=None, fig=1, sources='all', round_z=False, 
            AUV=None, wavelength=1600., sed_model=None, force_labels=False, **kwargs):
    
        return self.Plot(z=z, ax=ax, fig=fig, sources=sources, round_z=round_z,
            AUV=AUV, wavelength=1600, sed_model=None, quantity='smf', 
            force_labels=force_labels, **kwargs)              

    def PlotColors(self, pop, axes=None, fig=1, z_uvlf=[4,6,8,10], 
        z_beta=[4,5,6,7], sources='all', repeat_z=True, beta_phot=True, 
        show_Mstell=True, show_MUV=True, show_AUV=False, **kwargs):
        """
        Make a nice plot showing UVLF and UV CMD constraints and models.
        """
        
        if axes is None:
                        
            if show_Mstell:
                fig = pl.figure(tight_layout=False, figsize=(24, 6), num=fig)
                fig.subplots_adjust(left=0.1 ,right=0.9)
                gs = gridspec.GridSpec(4, 8, hspace=0.0, wspace=0.8, figure=fig)
                            
            else:
                fig = pl.figure(tight_layout=False, figsize=(12, 6), num=fig)
                fig.subplots_adjust(left=0.1 ,right=0.9)
                gs = gridspec.GridSpec(4, 4, hspace=0.0, wspace=0.05, figure=fig)
                
            if show_Mstell:
                ax_uvlf = fig.add_subplot(gs[:,0:2])
                ax_cmr4 = fig.add_subplot(gs[0,2:4])
                ax_cmr6 = fig.add_subplot(gs[1,2:4])
                ax_cmr8 = fig.add_subplot(gs[2,2:4])
                ax_cmr10 = fig.add_subplot(gs[3,2:4])
                
                ax_smf = fig.add_subplot(gs[:,4:6])
                ax_cMs4 = fig.add_subplot(gs[0,6:])
                ax_cMs6 = fig.add_subplot(gs[1,6:])
                ax_cMs8 = fig.add_subplot(gs[2,6:])
                ax_cMs10 = fig.add_subplot(gs[3,6:]) 
                
                ax_cMs = [ax_cMs4, ax_cMs6, ax_cMs8, ax_cMs10]       
            else:
                ax_uvlf = fig.add_subplot(gs[:,0:2])
                ax_cmr4 = fig.add_subplot(gs[0,2:])
                ax_cmr6 = fig.add_subplot(gs[1,2:])
                ax_cmr8 = fig.add_subplot(gs[2,2:])
                ax_cmr10 = fig.add_subplot(gs[3,2:])
                ax_cMs = []
                ax_smf = None
                
            ax_cmd = [ax_cmr4, ax_cmr6, ax_cmr8, ax_cmr10]

            axes = ax_uvlf, ax_cmd, ax_smf, ax_cMs
        
        else:
            ax_uvlf, ax_cmd, ax_smf, ax_cMs = axes
            ax_cmr4, ax_cmr6, ax_cmr8, ax_cmr10 = ax_cmd
            if show_Mstell:
                ax_cMs4, ax_cMs6, ax_cMs8, ax_cMs10 = ax_cMs
                
            
        l11 = read_lit('lee2011')
        b14 = read_lit('bouwens2014')
        f12 = read_lit('finkelstein2012')

        zall = np.sort(np.unique(np.concatenate((z_uvlf, z_beta))))
        colors = {4: 'k', 5: 'r', 6: 'b', 7: 'y', 8: 'c', 9: 'g', 10: 'm'}

        ##
        # Plot data
        ##
        mkw = {'capthick': 1, 'elinewidth': 1, 'alpha': 0.5, 'capsize': 4}
        
        ct_lf = 0
        ct_b = 0
        for j, z in enumerate(zall):

            if z in z_uvlf:
                _ax = self.PlotLF(z, ax=ax_uvlf, color=colors[z], mfc=colors[z],
                    mec=colors[z], sources=sources, round_z=0.21)
                ax_uvlf.annotate(r'$z \sim {}$'.format(z), (0.95, 0.25-0.05*ct_lf), 
                    xycoords='axes fraction', color=colors[z], ha='right', va='top')
        
                if show_Mstell:
                    _ax = self.PlotSMF(z, ax=ax_smf, color=colors[z], mfc=colors[z],
                        mec=colors[z], sources=sources, round_z=0.21)
                    ax_smf.annotate(r'$z \sim {}$'.format(z), (0.05, 0.25-0.05*ct_lf), 
                        xycoords='axes fraction', color=colors[z], ha='left', va='top')
        
                ct_lf += 1    
        
            if z not in z_beta:
                continue
                        
            if z in b14.data['beta']:
        
                err = b14.data['beta'][z]['err'] + b14.data['beta'][z]['sys']
                ax_cmd[j].errorbar(b14.data['beta'][z]['M'], b14.data['beta'][z]['beta'], 
                    yerr=err, 
                    fmt='o', color=colors[z], label=r'Bouwens+ 2014' if j == 0 else None,
                    **mkw)
                                                
            #if z in l11.data['beta']:
            #    ax_cmd[j].errorbar(l11.data['beta'][z]['M'], l11.data['beta'][z]['beta'], 
            #        l11.data['beta'][z]['err'], 
            #        fmt='*', color=colors[z], label=r'Lee+ 2011' if j == 0 else None,
            #        **mkw)
            
            ax_cmd[j].annotate(r'$z \sim {}$'.format(z), (0.95, 0.95), 
                ha='right', va='top', xycoords='axes fraction', color=colors[z])
            ct_b += 1
            
            if not show_Mstell:
                continue
                
            if z in f12.data['beta']:    
                err = f12.data['beta'][z]['err']
                ax_cMs[j].errorbar(10**f12.data['beta'][z]['Ms'], 
                    f12.data['beta'][z]['beta'], err.T, 
                    fmt='o', color=colors[z], 
                    label=r'Finkelstein+ 2012' if j == 0 else None,
                    **mkw)
                        
        ##
        # Plot models
        ##
        Ms = np.arange(6, 13.25, 0.25)
        mags = np.arange(-25, -12, 0.1)
        mags_cr = np.arange(-25, -10, 0.25)
        hst_shallow = b14.filt_shallow
        hst_deep = b14.filt_deep
        calzetti = read_lit('calzetti1994').windows
        
        for j, z in enumerate(zall):
            zstr = round(z)
            
            if z in z_uvlf:
                phi = pop.LuminosityFunction(z, mags)
    
                ax_uvlf.semilogy(mags, phi, color=colors[z], **kwargs)

                if show_Mstell:
                    phi = pop.StellarMassFunction(z, bins=Ms)
                    ax_smf.semilogy(10**Ms, phi, color=colors[z], **kwargs)    
            
            if z not in z_beta:
                continue
            
            if zstr >= 6:
                hst_filt = hst_deep
            else:
                hst_filt = hst_shallow

            cam = 'wfc', 'wfc3' if zstr <= 7 else 'nircam'    
            filt = hst_filt[zstr] if zstr <= 7 else None
            fset = None if zstr <= 7 else 'M'

            _beta_phot = pop.Beta(z, Mbins=mags_cr, return_binned=True,
                cam=cam, filters=filt, filter_set=fset, rest_wave=None,
                dlam=20.)
            _beta_spec = pop.Beta(z, Mbins=mags_cr, return_binned=True,
                    rest_wave=(1600., 3000.), dlam=20.)
            _mags = pop.Beta(z, Mbins=mags_cr, dlam=20.,
                cam=cam, filters=filt, filter_set=fset, rest_wave=None)
             
            if beta_phot:
                beta = _beta_phot
            else:
                beta = _beta_spec
                                                      
            ax_cmd[j].plot(mags_cr, beta, color=colors[z], **kwargs)
            
            if show_Mstell:
                
                _beta_c94 = pop.Beta(z, Mwave=1600., return_binned=False,
                    cam='calzetti', filters=calzetti, dlam=10., rest_wave=None)

                # Need to interpolate between Ms and MUV
                _Ms = pop.get_field(z, 'Ms')
                _nh = pop.get_field(z, 'nh')
                _x, _b, _err = bin_samples(np.log10(_Ms), _beta_c94, Ms, 
                    weights=_nh)
                
                ax_cMs[j].plot(10**_x, _b, color=colors[z], **kwargs)
                ax_cMs[j].annotate(r'$z \sim {}$'.format(z), (0.05, 0.95), 
                    ha='left', va='top', xycoords='axes fraction', color=colors[z])
                
            if repeat_z and j == 0:
                for k in range(1, 4):
                    ax_cmd[k].plot(mags_cr, beta, color=colors[z], **kwargs)
                    if show_Mstell:
                        ax_cMs[k].plot(10**Ms, _b, color=colors[z], **kwargs)
                                        
        ##
        # Clean-up
        ##
        for i, ax in enumerate([ax_uvlf] + ax_cmd):
            ax.set_xlim(-24, -15)
            ax.set_xticks(np.arange(-24, -15, 2))
            ax.set_xticks(np.arange(-24, -15, 1), minor=True)            
            
            if i > 0:
                ax.set_ylabel(r'$\beta$')
                ax.set_yticks(np.arange(-2.8, -0.8, 0.4))
                ax.set_yticks(np.arange(-2.9, -1., 0.1), minor=True)
                ax.set_ylim(-2.9, -1.)
                
                if not show_Mstell:
                    ax.yaxis.tick_right()
                    ax.yaxis.set_label_position("right")
                
                if i < 4:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel(r'$M_{\mathrm{UV}}$')
                    
                ax.yaxis.set_ticks_position('both')    
            else:
                ax.set_xlabel(r'$M_{\mathrm{UV}}$')
                ax.set_ylabel(labels['galaxy_lf'])
                ax.set_ylim(1e-7, 1e-1)
                            
        if show_Mstell:
            ax_smf.set_xlabel(r'$M_{\ast} / M_{\odot}$')
            ax_smf.set_ylabel(labels['galaxy_smf'])
            ax_smf.set_xscale('log')   
            ax_smf.set_ylim(1e-7, 1e-1)
            ax_smf.set_xlim(1e7, 1e12)
            for i, ax in enumerate([ax_cMs4, ax_cMs6, ax_cMs8, ax_cMs10]):     
                ax.set_xscale('log')
                ax.set_xlim(1e7, 1e11)
                ax.set_ylabel(r'$\beta$')
                ax.set_yticks(np.arange(-2.8, -0.8, 0.4))
                ax.set_yticks(np.arange(-2.9, -1., 0.1), minor=True)
                ax.set_ylim(-2.9, -1.)
                ax.yaxis.set_ticks_position('both')    
                if i < 3:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel(r'$M_{\ast} / M_{\odot}$')
        
        return ax_uvlf, ax_cmd, ax_smf, ax_cMs
        
    def PlotColorEvolution(self, pop, zarr=None, axes=None, fig=1, 
        wave_lo=1300., wave_hi=2600., which_nircam='W', **kwargs):
        """
        Plot Beta_19.5(z) and Beta_Mstell(z).
        """
        
        if axes is None:
            fig = pl.figure(tight_layout=False, figsize=(8, 8), num=fig)
            fig.subplots_adjust(left=0.1)
            gs = gridspec.GridSpec(2, 2, hspace=0.05, wspace=0.4, figure=fig)

            axB = fig.add_subplot(gs[0,0])
            axD = fig.add_subplot(gs[0,1])
            axB2 = fig.add_subplot(gs[1,0])
            axD2 = fig.add_subplot(gs[1,1])
        else:
            axB, axD, axB2, axD2 = axes

        # Plot the Bouwens data
        zbrack = [3.8, 5.0, 5.9, 7.0, 8.0]
        Beta195 = [-1.85, -1.91, -2.00, -2.05, -2.13]
        Beta195_err = [0.01, 0.02, 0.05, 0.09, 0.44]
        Beta195_sys = [0.06, 0.06, 0.08, 0.13, 0.27]

        dBdMUV = [-0.11, -0.14, -0.2, -0.2]
        dB_err = [0.01, 0.02, 0.04, 0.07]

        axB.errorbar(zbrack, Beta195, yerr=Beta195_sys, fmt='o', zorder=10,
            color='r')
        axD.errorbar(zbrack[:-1], dBdMUV, yerr=dB_err, fmt='o', zorder=10,
            color='r')

        mags = np.arange(-25, -10, 0.1)
        mags_cr = np.arange(-25.5, -10, 0.5)
        
        if zarr is None:
            zarr = np.arange(4, 12., 1.)

        linfunc = lambda x, p0, p1: p0 * (x - 8.) + p1
        cubfunc = lambda x, p0, p1, p2: p0 * (x - 8.)**2 + p1 * (x - 8.) + p2
        colors = 'r', 'y', 'gray'
        Mstell = np.array([7.5, 8.5, 9.5])
        f12 = read_lit('finkelstein2012')
        calzetti = read_lit('calzetti1994').windows
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
                    
                axB2.errorbar(z, y, yerr=yerr, fmt='o', color=colors[i],
                    label=lab)
        
            axD2.errorbar(z, f12.data['slope_wrt_mass'][z]['slope'],
                yerr=f12.data['slope_wrt_mass'][z]['err'],
                color=colors[i], fmt='o')
                    
            #sig = np.mean(f12.data['beta'][z]['err'], axis=1)
            #popt, pcov = curve_fit(linfunc, Mstell, f12.data['beta'][z]['beta'], 
            #    sigma=sig, p0=[0.3, 0.], maxfev=1000)
            #popt2, pcov2 = curve_fit(cubfunc, Mstell, f12.data['beta'][z]['beta'], 
            #    p0=[0.0, 0.3, 0.], maxfev=1000)
            #cubrecon = popt2[0] * (Mstell - 8.)**2 + popt2[1] * Mstell + popt2[2]
            #
            #cubeder = 2 * popt2[0] * (Mstell - 8.) + popt2[1]
            #
            #s2 = np.interp(8., Mstell, cubeder)
            #axD2.errorbar(z, s2, color='r', fmt='o')
            #print('add errors to me')
            

        # For CANDELS, ERS    
        b14 = read_lit('bouwens2014')
        hst_shallow = b14.filt_shallow
        hst_deep = b14.filt_deep

        nircam = Survey(cam='nircam')
        nircam_M = nircam._read_nircam(filter_set='M')
        nircam_W = nircam._read_nircam(filter_set='W')

        colors = {4: 'k', 5: 'r', 6: 'b', 7: 'y', 8: 'c', 9: 
            'g', 10: 'm', 11: 'gray'}

        ##
        # Loop over models and reconstruct best-fitting Beta(z).
        ##
        Ms_b = np.arange(6.5, 11., 0.5)
        colors = 'k', 'k', 'k', 'k'
        ls = '-', '--', ':'

        ##
        # Won't be able to do DerivedBlob for 'nozevo' case because we only
        # saved at one redshift :( Will be crude for others. Could re-generate
        # samples later (parallelize, on cluster).
        ##
        _colors = {4: 'k', 5: 'r', 6: 'b', 7: 'y', 8: 'c', 9: 'g', 10: 'm'}
        mkw = {'capthick': 1, 'elinewidth': 1, 'alpha': 0.5, 'capsize': 4}    
        
        print("Computing UV slope evolution for model={}...".format(i))
    
        B195_hst = []
        dBdM195_hst = []
        B195_spec = []
        dBdM195_spec = []
        B195_jwst = []
        dBdM195_jwst = []
        BMstell = []
        dBMstell = []
        for j, z in enumerate(zarr):
        
            zstr = round(z)
        
            if zstr >= 6:
                hst_filt = hst_deep
            else:
                hst_filt = hst_shallow
        
            cam = ('wfc', 'wfc3') if zstr <= 7 else ('nircam', )
            filt = hst_filt[zstr] if zstr <= 7 else None
            fset = None if zstr <= 7 else 'M'        
        
            beta_spec = pop.Beta(z, Mbins=mags_cr, return_binned=True,
                rest_wave=(wave_lo, wave_hi))
            beta_hst = pop.Beta(z, Mbins=mags_cr, return_binned=True,
                cam=cam, filters=filt, filter_set=fset, rest_wave=None) 
        
            # Compute raw beta and compare to Mstell    
            beta_c94 = pop.Beta(z, Mwave=1600., return_binned=False,
                cam='calzetti', filters=calzetti, dlam=10., rest_wave=None)

            # Compute beta(Mstell)
            beta_Mst = pop.Beta(z, Mwave=1600., return_binned=False,
                cam='calzetti', filters=calzetti, dlam=10., rest_wave=None,
                Mstell=10**Ms_b, massbins=Ms_b)

            #Ms_r = pop.get_field(z, 'Ms')
            #nh_r = pop.get_field(z, 'nh')
            #
            #_x1, _y1, _err = bin_samples(np.log10(Ms_r), beta_c94, Ms_b, 
            #    weights=nh_r)    
            #
            #_tmp = []  
            #for Mstell in [7.5, 8.5, 9.5]:
            #    s1 = np.interp(Mstell-0.5, Ms_b, _y1)
            #    s2 = np.interp(Mstell+0.5, Ms_b, _y1)
            #    _tmp.append(np.mean([s1, s2]))
        
            # Compute slopes with Mstell
            #popt, pcov = curve_fit(linfunc, Ms_b, beta_Mst, p0=[0.3, 0.], 
            #    maxfev=1000)
            #popt2, pcov2 = curve_fit(cubfunc, Ms_b, beta_Mst, p0=[0.0, 0.3, 0.], 
            #    maxfev=1000)
            #cubrecon = popt2[0] * (Ms_b - 8.)**2 + popt2[1] * Ms_b + popt2[2]
            #cubeder = 2 * popt2[0] * (Ms_b - 8.) + popt2[1]
                 
            BMstell.append(beta_Mst)
            dBMstell.append(pop.dBeta_dMstell(z, Mstell=10**Ms_b, massbins=Ms_b))
                    
            # Compute beta given HST+JWST
            cam2 = ('wfc', 'wfc3', 'nircam') if zstr <= 7 else ('nircam', )
            filt2 = hst_filt[zstr] if zstr <= 7 else None
            # Add JWST filters based on redshift?
            if filt2 is not None:
                now = list(filt2)
            else:
                now = []
        
            nircam_z = what_filters(z, 
                nircam_M if which_nircam=='M' else nircam_W, wave_lo, wave_hi)
            print("Added NIRCAM at z={}: {}".format(z, nircam_z))
            now.extend(nircam_z)
        
            filt2 = tuple(now)
                
            beta_jwst = pop.Beta(z, Mbins=mags_cr, return_binned=True,
                cam=cam2, filters=filt2, filter_set=fset, 
                rest_wave=None)
        
            # Compute Beta at MUV=-19.5
            for k, beta in enumerate([beta_spec, beta_hst, beta_jwst]):
        
                #if k == 0:
                #    continue
        
                _i195 = np.argmin(np.abs(mags_cr + 19.5))
                _B195 = beta[_i195]
        
                # Compute dBeta/dMag via finite difference.
                #_xx = mags[_i195-3:_i195+4]
                #_yy = beta[_i195-3:_i195+4]
                #
                #xx, yy = central_difference(_xx, _yy)
                #
                ## Smooth this out by just using last two points
                #slope = np.interp(-19.5, [xx[0], xx[-1]], [yy[0], yy[-1]])

                # Compute dBeta/dMag by fitting PL to points.
                _xx = mags_cr[_i195-2:_i195+2]
                _yy = beta[_i195-2:_i195+2]
        
                if not np.any(np.isfinite(_yy)):
                    if k == 0:
                        B195_spec.append(-99999)
                        dBdM195_spec.append(-99999)
                    elif k == 1:
                        B195_hst.append(-99999)
                        dBdM195_hst.append(-99999)
                    else:
                        B195_jwst.append(-99999)   
                        dBdM195_jwst.append(-99999) 
            
                    continue
        
                func = lambda xx, p0, p1: p0 + p1 * xx
                popt, pcov = curve_fit(func, _xx, _yy, p0=np.array([-2., 0.]))

                norm = popt[0]
                slope = popt[1]
        
                if k == 0:
                    B195_spec.append(_B195)
                    dBdM195_spec.append(slope)
                elif k == 1:
                    B195_hst.append(_B195)
                    dBdM195_hst.append(slope)
                else:
                    B195_jwst.append(_B195)
                    dBdM195_jwst.append(slope)

        ##
        # Finish up and plot.
        ##
        B195_spec = np.array(B195_spec)        
        dBdM195_spec = np.array(dBdM195_spec)
        ok_spec = B195_spec > -99999
        axB.plot(zarr[ok_spec==1], B195_spec[ok_spec==1], lw=1, alpha=0.4,
            **kwargs)
        axD.plot(zarr[ok_spec==1], dBdM195_spec[ok_spec==1], lw=1, alpha=0.4,
            **kwargs)
    
        B195_hst = np.array(B195_hst)        
        dBdM195_hst = np.array(dBdM195_hst)
        ok_hst = B195_hst > -99999
        axB.plot(zarr[ok_hst==1], B195_hst[ok_hst==1], lw=2, **kwargs)
        axD.plot(zarr[ok_hst==1], dBdM195_hst[ok_hst==1], lw=2, **kwargs)
    
        B195_jwst = np.array(B195_jwst)        
        dBdM195_jwst = np.array(dBdM195_jwst)
        ok_jwst = B195_jwst > -99999
        axB.plot(zarr[ok_jwst==1], B195_jwst[ok_jwst==1], lw=5, alpha=0.4, 
            **kwargs)
        axB.plot(zarr[ok_jwst==1], B195_jwst[ok_jwst==1], lw=1, alpha=1, 
            **kwargs)
        axD.plot(zarr[ok_jwst==1], dBdM195_jwst[ok_jwst==1], lw=5, alpha=0.4, 
            **kwargs)
        axD.plot(zarr[ok_jwst==1], dBdM195_jwst[ok_jwst==1], lw=1, alpha=1,
            **kwargs)
        
        ##
        # Plot Mstell stuff
        ##
        for logM in [7, 8, 9, 10]:
            j = np.argmin(np.abs(Ms_b - logM))
            axB2.plot(zarr, np.array(BMstell)[:,j], **kwargs)    
            axD2.plot(zarr, np.array(dBMstell)[:,j], **kwargs)
        
        ##
        # Clean up
        ##
        axD.set_yticks(np.arange(-0.3, 0, 0.1))
        axD.set_yticks(np.arange(-0.3, 0, 0.05), minor=True)
        
        axB.set_ylim(-2.9, -1.3)
        axB.set_xlim(3.5, 11.2)
        axD.set_xlim(3.5, 11.2)
        axD.set_ylim(-0.3, -0.05)
        #axB.set_xticklabels([])
        axB2.set_ylim(-2.9, -1.3)
        axB2.set_xlim(3.5, 11.2)
        axD2.set_xlim(3.5, 11.2)
        axD2.set_ylim(0., 0.5)

        axB.set_xticklabels([])
        axD.set_xticklabels([])
        
        axB.yaxis.set_ticks_position('both')
        axB2.yaxis.set_ticks_position('both')
        axD.yaxis.set_ticks_position('both')
        axD2.yaxis.set_ticks_position('both')

        axB.set_yticks(np.arange(-3, -1.3, 0.1), minor=True)
        axB2.set_yticks(np.arange(-3, -1.3, 0.1), minor=True)

        axB2.legend(loc='upper left', frameon=True, fontsize=8)

        if axes is None:
            axB.set_ylabel(r'$\beta(M_\mathrm{UV}=-19.5)$')
            axB2.set_ylabel(r'$\beta(\log_{10}M_{\ast})$')
            axD.set_ylabel(r'$d\beta(M_\mathrm{UV}=-19.5)/dM_{\mathrm{UV}}$')
            axD2.set_ylabel(r'$d\beta(\log_{10}M_{\ast})/dlog_{10}M_{\ast}$')
            axD2.set_xlabel(r'$z$')
            axB2.set_xlabel(r'$z$')
        
        for ax in [axB, axD, axB2, axD2]:
            ax.yaxis.set_label_coords(-0.15, 0.5)
            ax.yaxis.set_label_coords(-0.15, 0.5)
        
        return axB, axD, axB2, axD2
        
    def Plot(self, z, ax=None, fig=1, sources='all', round_z=False, force_labels=False,
        AUV=None, wavelength=1600., sed_model=None, quantity='lf', use_labels=True,
        take_log=False, imf=None, mags='intrinsic', sources_except=[], **kwargs):
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
                                        
            M = np.array(data[source]['M'])
            phi = np.array(data[source]['phi'])
            err = np.array(data[source]['err'])
            ulim = np.array(data[source]['ulim'])

            kw = {'fmt':'o', 'ms':5, 'elinewidth':2, 'mew': 2, 
                'mec':default_colors[source],
                'fmt': default_markers[source],
                'color':default_colors[source], 'capthick':2}
            
            if not use_labels:
                label = None
            elif ('label' not in kwargs):
                label = source
            else:
                label = kwargs['label']
            
            kw['label'] = label
            kw.update(kwargs)
                
            if AUV is not None:
                dc = AUV(z, np.array(M))
            else:
                dc = 0
                
            # Shift band [optional]
            if quantity in ['lf']:
                if data[source]['wavelength'] != wavelength:
                    #shift = sed_model.
                    print("WARNING: {0!s} wavelength={1}A, not {2}A!".format(\
                        source, data[source]['wavelength'], wavelength))
            #else:
            shift = 0.    
                          
            ax.errorbar(M+shift-dc, phi, yerr=err, uplims=ulim, zorder=10, 
                **kw)

        if quantity == 'lf' and ((not gotax) or force_labels):
            ax.set_xticks(np.arange(-26, 0, 1), minor=True)
            ax.set_xlim(-26.5, -10)
            ax.set_xlabel(r'$M_{\mathrm{UV}}$')    
            ax.set_ylabel(r'$\phi(M_{\mathrm{UV}}) \ [\mathrm{mag}^{-1} \ \mathrm{cMpc}^{-3}]$')
        elif quantity == 'smf' and ((not gotax) or force_labels):
            ax.set_xscale('log')
            ax.set_xlim(1e7, 1e13)
            ax.set_xlabel(r'$M_{\ast} / M_{\odot}$')    
            ax.set_ylabel(r'$\phi(M_{\ast}) \ [\mathrm{dex}^{-1} \ \mathrm{cMpc}^{-3}]$')
        elif quantity == 'mzr' and ((not gotax) or force_labels):
            ax.set_xlabel(r'$M_{\ast} / M_{\odot}$')
            ax.set_ylabel(r'$12+\log{\mathrm{O/H}}$')
            ax.set_xlim(1e8, 1e12)
            ax.set_ylim(7, 9.5)
            
        if quantity in ['lf', 'smf']:
            ax.set_yscale('log', nonposy='clip')    
            
        pl.draw()
        
        return ax
            
    def MultiPlot(self, redshifts, sources='all', round_z=False, ncols=1, 
        panel_size=(0.75,0.75), fig=1, xmax=-10, ymax=10, legends=None, AUV=None,
        quantity='lf', mp=None, sources_except=[], 
        mp_kwargs={}, show_ylabel=True, **kwargs):
        """
        Plot the luminosity function at a bunch of different redshifts.
        
        Parameters
        ----------
        z : list
            List of redshifts to include.
        ncols : int
            How many columns in multiplot? Number of rows will be determined
            automatically.
        legends : bool, str
            'individual' means one legend per axis, 'master' means one
            (potentially gigantic) legend.
            
        """        
        
        if ncols == 1:
            nrows = len(redshifts)
        else:
            nrows = len(redshifts) // ncols
            
        if nrows * ncols != len(redshifts):
            nrows += 1
            
        dims = (nrows, ncols)    
            
        # Force redshifts to be in ascending order
        if not np.all(np.diff(redshifts)) > 0:   
            redshifts = np.sort(redshifts)
            
        if mp_kwargs == {}:
            mp_kwargs = {'panel_size': panel_size, 'padding': [0.2]*2}
            
        annotate_z = 'left' if quantity == 'lf' else 'right'
            
        # Create multiplot
        if mp is None:
            gotmp = False
            mp = MultiPanel(dims=dims, fig=fig, **mp_kwargs)
        else:
            gotmp = True
            assert mp.dims == dims
        
        if not hasattr(self, 'redshifts_in_mp'):
            self.redshifts_in_mp = {}
        
        if quantity not in self.redshifts_in_mp:
            self.redshifts_in_mp[quantity] = []
        
        for i, z in enumerate(redshifts):
            k = mp.elements.ravel()[i]
            ax = mp.grid[k]
            
            # Where in the MultiPlot grid are we?
            self.redshifts_in_mp[quantity].append(k)
                        
            self.Plot(z, sources=sources, round_z=round_z, ax=ax, AUV=AUV,
                quantity=quantity, sources_except=sources_except, **kwargs)
            
            if annotate_z == 'left':
                _xannot = 0.05
            else:
                _xannot = 0.95
                
            if gotmp:
                continue
                
            ax.annotate(r'$z \sim {}$'.format(round(z, 1)), (_xannot, 0.95), 
                ha=annotate_z, va='top', xycoords='axes fraction')
        
        if gotmp:
            return mp
                        
        for i, z in enumerate(redshifts):
            k = mp.elements.ravel()[i]
            ax = mp.grid[k]
            
            if quantity == 'lf':
                ax.set_xlim(-24, xmax)
                ax.set_ylim(1e-7, ymax)
                ax.set_yscale('log', nonposy='clip')  
                ax.set_ylabel('')
                ax.set_xlabel(r'$M_{\mathrm{UV}}$')
            else:
                ax.set_xscale('log')
                ax.set_xlim(1e6, 1e12)
                ax.set_ylim(1e-7, ymax)
                ax.set_yscale('log', nonposy='clip')                      
                ax.set_xlabel(r'$M_{\ast} / M_{\odot}$')
        
        if show_ylabel:
            if quantity == 'lf':
                mp.global_ylabel(r'$\phi(M_{\mathrm{UV}}) \ [\mathrm{mag}^{-1} \ \mathrm{cMpc}^{-3}]$')
            else:
                mp.global_ylabel(r'$\phi(M_{\ast}) \ [\mathrm{dex}^{-1} \ \mathrm{cMpc}^{-3}]$')
        
            
        pl.show()    
            
        return mp
        
    def PlotScalingRelations(self, include=['SMHM', 'MZR', 'MS'], ncols=None):
        """
        
        """
        pass
        
    def PlotTrajectories(self):
        pass
            
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
        
    def add_master_legend(self, mp, **kwargs):
        return add_master_legend(mp, **kwargs)
        
    def MegaPlot(self, pop, axes=None, fig=1, use_best=True, method='mode',
        fresh=False, **kwargs):
        """
        Make a huge plot.
        """
        
        if axes is None:
            gotax = False
            axes = self._MegaPlotSetup(fig)
        else:
            gotax = True

        if not gotax:
            self._MegaPlotCalData(axes)
            self._MegaPlotPredData(axes)
            self._MegaPlotGuideEye(axes)

        if isinstance(pop, GalaxyEnsemble):
            self._MegaPlotPop(axes, pop)
        elif hasattr(pop, 'chain'):
            if fresh:
                bkw = pop.base_kwargs.copy()
                bkw.update(pop.max_likelihood_parameters(method=method))
                pop = GP(**bkw)
                self._MegaPlotPop(axes, pop)
            else:
                self._MegaPlotChain(axes, pop, use_best=use_best, **kwargs)
        else:
            raise TypeError("Unrecognized object pop={}".format(pop))
         
         
        self._MegaPlotCleanup(axes)
        
        return axes
        
    def _MegaPlotPop(self, kw, pop, **kwargs):
        
        
        ax_sfe = kw['ax_sfe']
        ax_fco = kw['ax_fco']
        ax_rdu = kw['ax_rdu']
        ax_phi = kw['ax_phi']
        ax_bet = kw['ax_bet']        
        
        ax_smf    = kw['ax_smf']
        ax_smhm   = kw['ax_smhm']
        ax_MsMUV  = kw['ax_MsMUV']
        ax_AUV    = kw['ax_AUV']
        ax_sfrd   = kw['ax_sfrd']
        ax_lae_z  = kw['ax_lae_z']
        ax_lae_m  = kw['ax_lae_m']
        ax_sfms   = kw['ax_sfms']
        
        _mst  = np.arange(6, 12, 0.2)
        _mags = np.arange(-25, -10, 0.2)
        
        redshifts = [4, 6, 8, 10]
        colors = ['k', 'b', 'c', 'm']
        
        dc1 = DustCorrection(dustcorr_method='meurer1999',
            dustcorr_beta='bouwens2014')
        
        xa_b = []
        xa_f = []
        for j, z in enumerate(redshifts):
            
            # UVLF
            phi = pop.LuminosityFunction(z, _mags)
            ax_phi.semilogy(_mags, phi, color=colors[j], drawstyle='steps-mid')
                    
            # Binned version
            Mbins = np.arange(-25, -10, 1.0)
            _beta = pop.Beta(z, Mwave=1600, return_binned=True,
                Mbins=Mbins)
            
            Mh = pop.get_field(z, 'Mh')
            Ms = pop.get_field(z, 'Ms')
            SFR = pop.get_field(z, 'SFR')
            SFE = pop.guide.SFE(z=z, Mh=Mh)
            
            ax_sfe.loglog(Mh, SFE, color=colors[j], alpha=0.8,
                label=r'$z={}$'.format(z))
                
            if pop.pf['pop_scatter_mar'] > 0:
                _bins = np.arange(7, 12.1, 0.1)
                x, y, std = bin_samples(np.log10(Ms), np.log10(SFR), _bins)
                ax_sfms.loglog(10**x, 10**y, color=colors[j])
            else:    
                ax_sfms.loglog(Ms, SFR, color=colors[j])
            
            # SMF
            phi = pop.StellarMassFunction(z, _mst)
            ax_smf.loglog(10**_mst, phi, color=colors[j], drawstyle='steps-mid')

            # SMHM
            _Mh = 10**np.arange(8, 12.5, 0.1)
            fstar = pop.SMHM(z, _Mh, return_mean_only=True)
            ax_smhm.loglog(_Mh, 10**fstar, color=colors[j])
            
            mags = pop.Magnitude(z, wave=1600.)
            beta = pop.Beta(z, Mwave=1600., return_binned=False)
            
            # MUV-Mstell
            _x, _y, _z = bin_samples(mags, np.log10(Ms), Mbins)
            ax_MsMUV.plot(_x, _y, color=colors[j])    
            
            # Beta just to get 'mags'
            if pop.pf['pop_dust_yield'] == 0:
                xa_f.append(0)
                xa_b.append(0)
                
                ax_bet.plot(Mbins, dc1.Beta(z, Mbins), color=colors[j])
                
                continue
                
            ax_bet.plot(Mbins, _beta, color=colors[j])    
            
            fcov = pop.guide.dust_fcov(z=z, Mh=Mh)
            Rdust = pop.guide.dust_scale(z=z, Mh=Mh)
            ydust = pop.guide.dust_yield(z=z, Mh=Mh)
            
            if pop.pf['pop_fduty'] is not None:
                fduty = pop.guide.fduty(z=z, Mh=Mh)
            else:
                fduty = np.zeros_like(Mh)
            
            if type(fcov) in [int, float, np.float64]:
                fcov = fcov * np.ones_like(Mh)
            
            #any_fcov = np.any(np.diff(fcov, axis=1) != 0)
            #any_fduty = np.any(np.diff(fduty, axis=1) != 0)
                        
            #if not np.all(np.diff(fcov) == 0):
            #    ax_fco.semilogx(Mh, fcov, color=colors[j])
            #    ax_fco.set_ylabel(r'$f_{\mathrm{cov}}$')
            #elif not np.all(np.diff(ydust) == 0):
            #    ax_fco.semilogx(Mh, ydust, color=colors[j])
            #    ax_fco.set_ylabel(r'$y_{\mathrm{dust}}$')
            #elif not np.all(np.diff(fduty) == 0):
            #    ax_fco.semilogx(Mh, fduty, color=colors[j])
            #    ax_fco.set_ylabel(r'$f_{\mathrm{duty}}$')
                
            ax_rdu.loglog(Mh, Rdust, color=colors[j])

            Mbins = np.arange(-25, -10, 1.)
            AUV = pop.AUV(z, Mwave=1600., return_binned=True,
                magbins=Mbins)
            
            ax_AUV.plot(Mbins, AUV, color=colors[j])
                            
            # LAE stuff
            _x, _y, _z = bin_samples(mags, fcov, Mbins)
            ax_lae_m.plot(_x, 1. - _y, color=colors[j])
            
            faint  = np.logical_and(Mbins >= -20.25, Mbins < -18.)
            bright = Mbins < -20.25
            
            xa_f.append(1. - np.mean(_y[faint==1]))    
            xa_b.append(1. - np.mean(_y[bright==1]))
            
        ax_lae_z.plot(redshifts, xa_b, color='k', alpha=1.0, ls='-')
        ax_lae_z.plot(redshifts, xa_f, color='k', alpha=1.0, ls='--')
        
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
        ax_smhm   = kw['ax_smhm']
        ax_MsMUV  = kw['ax_MsMUV']
        ax_AUV    = kw['ax_AUV']
        ax_sfrd   = kw['ax_sfrd']
        ax_lae_z  = kw['ax_lae_z']
        ax_lae_m  = kw['ax_lae_m']
        ax_sfms   = kw['ax_sfms']
        
        _mst  = np.arange(6, 12, 0.2)
        _mags = np.arange(-25, -10, 0.2)

        redshifts = [4, 6, 8, 10]
        colors = ['k', 'b', 'c', 'm']

        dc1 = DustCorrection(dustcorr_method='meurer1999',
            dustcorr_beta='bouwens2014')
            
            
        # Compute X_LAE, etc.
        if 'dust_fcov' in anl.all_blob_names:
            func = lambda data, ivars: 1. - data['dust_fcov']
            anl.DeriveBlob(name='x_LAE', func=func, clobber=True,
                fields='dust_fcov', ivar=None) 
                              
        xa_b = []
        xa_f = []
        for j, z in enumerate(redshifts):
            
            # UVLF
            anl.ReconstructedFunction('galaxy_lf', ivar=[z, None], ax=ax_phi,
                color=colors[j], **kwargs)
                
            anl.ReconstructedFunction('fstar', ivar=[z, None], ax=ax_sfe,
                color=colors[j], **kwargs)    
        
            anl.ReconstructedFunction('galaxy_smf', ivar=[z, None], ax=ax_smf,
                color=colors[j], is_logx=True, **kwargs)
            
            #if 'MUV_gm' in anl.all_blob_namess:
            #    _z, _MUV = anl.get_ivars('MUV_gm')
            #    k = np.argmin(np.abs(z - _z))
            #    new_x = anl.ExtractData('MUV_gm')['MUV_gm'][:,k,:]
            #    print("New magnitudes!!!!")
            #else:
            new_x = None
            
            anl.ReconstructedFunction('beta_hst', ivar=[z, None], ax=ax_bet,
                color=colors[j], new_x=new_x, **kwargs)
            
            if 'use_best' in kwargs:
                if kwargs['use_best']:
                    anl.ReconstructedFunction('beta_spec', ivar=[z, None], ax=ax_bet,
                        color=colors[j], ls='--', lw=3, **kwargs)    
            
            anl.ReconstructedFunction('AUV', ivar=[z, None], ax=ax_AUV,
                color=colors[j], **kwargs)
                
            anl.ReconstructedFunction('sfrd', ivar=None, ax=ax_sfrd,
                color=colors[j], **kwargs)
        
            anl.ReconstructedFunction('dust_scale', ivar=[z, None], ax=ax_rdu,
                color=colors[j], **kwargs)
            
            if 'fduty' in anl.all_blob_names:
                anl.ReconstructedFunction('fduty', ivar=[z, None], ax=ax_fco,
                    color=colors[j], **kwargs)
            
            if 'dust_fcov' in anl.all_blob_names:
                anl.ReconstructedFunction('dust_fcov', ivar=[z, None], ax=ax_fco,
                    color=colors[j], **kwargs)    
            
                # Need to convert to MUV!
                anl.ReconstructedFunction('x_LAE', ivar=[z, None], ax=ax_lae_m,
                    color=colors[j], **kwargs)
                
                    
                
                
    def _MegaPlotLimitsAndTicks(self, kw):
        ax_sfe = kw['ax_sfe']
        ax_fco = kw['ax_fco']
        ax_rdu = kw['ax_rdu']
        ax_phi = kw['ax_phi']
        ax_bet = kw['ax_bet']

        ax_smf    = kw['ax_smf']
        ax_smhm   = kw['ax_smhm']
        ax_MsMUV  = kw['ax_MsMUV']
        ax_AUV    = kw['ax_AUV']
        ax_sfrd   = kw['ax_sfrd']
        ax_lae_z  = kw['ax_lae_z']
        ax_lae_m  = kw['ax_lae_m']
        ax_sfms   = kw['ax_sfms']
        
        
        
        ax_sfe.set_xlim(1e8, 1e13)
        ax_sfe.set_ylim(1e-3, 1.0)
        ax_fco.set_xscale('log')
        ax_fco.set_xlim(1e8, 1e13)
        ax_fco.set_yscale('linear')
        ax_fco.set_ylim(0, 1.05)
        ax_rdu.set_xlim(1e8, 1e13)
        ax_rdu.set_ylim(1e-2, 100)
        
        ax_smf.set_xscale('log')
        ax_smf.set_xlim(1e7, 1e12)
        ax_smf.set_ylim(1e-7, 2e-1)
        ax_smhm.set_xscale('log')
        ax_smhm.set_yscale('log')
        #ax_smhm.set_ylim(-4, 1.)
        #ax_smhm.set_yscale('log', nonposy='clip')
        ax_smhm.set_xlim(1e9, 1e12)
        ax_smhm.set_ylim(5e-4, 1.5e-1)
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
        
        ax_lae_m.set_xlim(-25, -12)
        ax_lae_z.set_xlim(3., 7.2)
        ax_lae_m.set_ylim(-0.05, 1.05)
        ax_lae_z.set_ylim(-0.05, 1.05)

        ax_sfrd.set_yscale('log')
        ax_sfrd.set_ylim(1e-4, 1e-1)

        # Set ticks for all MUV scales
        for ax in [ax_bet, ax_phi, ax_MsMUV, ax_lae_m, ax_AUV]:
            ax.set_xticks(np.arange(-24, -12, 1), minor=True)
            
        for ax in [ax_MsMUV, ax_lae_m, ax_AUV]:
            ax.set_xlim(-25, -15)    
        
        return kw
        
    def _MegaPlotSetup(self, fig):
        
        fig = pl.figure(tight_layout=False, figsize=(22, 7), num=fig)
        #gs = gridspec.GridSpec(3, 10, hspace=0.3, wspace=1.0)
        gs = gridspec.GridSpec(3, 14, hspace=0.3, wspace=5.0)
        
        # Inputs
        ax_sfe = fig.add_subplot(gs[0,0:3])
        ax_fco = fig.add_subplot(gs[1,0:3])
        ax_rdu = fig.add_subplot(gs[2,0:3])
        
        # Predictions
        ax_smf = fig.add_subplot(gs[0:2,6:9])
        ax_smhm = fig.add_subplot(gs[2,12:])
        ax_MsMUV = fig.add_subplot(gs[2,9:12])
        ax_AUV = fig.add_subplot(gs[0,9:12])
        ax_sfrd = fig.add_subplot(gs[0,12:])
        ax_lae_z = fig.add_subplot(gs[1,12:])
        ax_lae_m = fig.add_subplot(gs[1,9:12])
        ax_sfms = fig.add_subplot(gs[2,6:9])
        
        # Cal
        ax_phi = fig.add_subplot(gs[0:2,3:6])
        ax_bet = fig.add_subplot(gs[2,3:6])

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
         'ax_smhm': ax_smhm,
         'ax_MsMUV': ax_MsMUV,
         'ax_AUV': ax_AUV, 
         'ax_sfrd': ax_sfrd,
         'ax_lae_z': ax_lae_z,
         'ax_lae_m': ax_lae_m,
         'ax_sfms': ax_sfms,
        }
        
        return kw
           
    def _MegaPlotCalData(self, kw):
        
        ax_sfe = kw['ax_sfe']
        ax_fco = kw['ax_fco']
        ax_rdu = kw['ax_rdu']
        ax_phi = kw['ax_phi']
        ax_bet = kw['ax_bet']
        
        
        ax_smf    = kw['ax_smf']
        ax_smhm   = kw['ax_smhm']
        ax_MsMUV  = kw['ax_MsMUV']
        ax_AUV    = kw['ax_AUV']
        ax_sfrd   = kw['ax_sfrd']
        ax_lae_z  = kw['ax_lae_z']
        ax_lae_m  = kw['ax_lae_m']
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
        redshifts = [4, 6, 8, 10]
        colors = 'k', 'b', 'c', 'm'
        mkw = {'capthick': 1, 'elinewidth': 1, 'alpha': 0.5, 'capsize': 4}

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
                round_z=0.1, color=colors[j], mec=colors[j], mfc=colors[j], mew=1, fmt='o',
                label='Song+ 2016' if j == 0 else None, **mkw)    
            self.PlotSMF(z, ax=ax_smf, sources=['stefanon2017'], mew=1, fmt='s',
                round_z=0.1, color=colors[j], mec=colors[j], mfc='none',
                label='Stefanon+ 2017' if j == 0 else None, **mkw)

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
                
    def _MegaPlotGuideEye(self, kw):
        ax_sfe = kw['ax_sfe']
        ax_fco = kw['ax_fco']
        ax_rdu = kw['ax_rdu']
        ax_phi = kw['ax_phi']
        ax_bet = kw['ax_bet']
        
        
        ax_smf    = kw['ax_smf']
        ax_smhm   = kw['ax_smhm']
        ax_MsMUV  = kw['ax_MsMUV']
        ax_AUV    = kw['ax_AUV']
        ax_sfrd   = kw['ax_sfrd']
        ax_lae_z  = kw['ax_lae_z']
        ax_lae_m  = kw['ax_lae_m']
        ax_sfms   = kw['ax_sfms']
        
        ax_rdu.annotate(r'$R_h \propto M_h^{1/3} (1+z)^{-1}$', (1.5e8, 30))
        
        redshifts = [4, 6, 8, 10]
        colors = 'k', 'b', 'c', 'm'
        
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

        mh = np.logspace(7., 8, 50.)
        ax_sfms.loglog(mh, 200 * func(4., 3./3.), 
            color=colors[0], lw=1, ls='-', alpha=0.5)    
        ax_sfms.annotate(r'$1$',   (mh[-1]*1.1, 200 * func(4., 3./3.)[-1]), ha='left')
                
    def _MegaPlotPredData(self, kw):
        
        
        ax_sfe = kw['ax_sfe']
        ax_fco = kw['ax_fco']
        ax_rdu = kw['ax_rdu']
        ax_phi = kw['ax_phi']
        ax_bet = kw['ax_bet']
        
        
        ax_smf    = kw['ax_smf']
        ax_smhm   = kw['ax_smhm']
        ax_MsMUV  = kw['ax_MsMUV']
        ax_AUV    = kw['ax_AUV']
        ax_sfrd   = kw['ax_sfrd']
        ax_lae_z  = kw['ax_lae_z']
        ax_lae_m  = kw['ax_lae_m']
        ax_sfms   = kw['ax_sfms']
        
        
        mkw = {'capthick': 1, 'elinewidth': 1, 'alpha': 0.5, 'capsize': 4}
        
        redshifts = [4, 6, 8, 10]
        colors = 'k', 'b', 'c', 'm'
        
        xarr = np.arange(-22, -18, 0.5)
        yarr = [0.1, 0.08, 0.08, 0.1, 0.18, 0.3, 0.47, 0.6]
        yerr = [0.1, 0.05, 0.03, 0.05, 0.05, 0.1, 0.15, 0.2]
        ax_lae_m.errorbar(xarr, yarr, yerr=yerr, color='k', 
            label='Stark+ 2010 (3 < z < 6.2)', fmt='o', **mkw)

        zlist = [4., 5, 6.1]
        x25_b = [0.13, 0.25, 0.2]
        x25_f = [0.35, 0.48, 0.55]
        err_b = [0.05, 0.05, 0.08]
        err_f = [0.05, 0.1, 0.15]
        
        _colors = 'k', 'g', 'b'
        for j, z in enumerate(zlist):
            ax_lae_z.errorbar(zlist[j], x25_b[j], yerr=err_b[j], 
                color=_colors[j], ms=5, 
                label=r'Stark+ 2011' if j == 0 else None,
                fmt='s', mfc='none', **mkw)
            ax_lae_z.errorbar(zlist[j], x25_f[j], yerr=err_f[j],
                color=_colors[j], ms=5,
                fmt='o', mfc='none', **mkw)

    
        # De Barros et al. (2017)    
        ax_lae_z.errorbar(5.9, 0.1, 0.05, color='b', fmt='*', mfc='none', ms=5,
            label=r'deBarros+ 2017', **mkw)
        ax_lae_z.errorbar(5.9, 0.38, 0.12, color='b', fmt='*', mfc='none', ms=5,
            **mkw)

        ax_lae_z.legend(loc='upper left', frameon=True, fontsize=6)
        ax_lae_m.legend(loc='upper left', frameon=True, fontsize=6)

        # Salmon et al. 2015
        data = \
        {
         4: {'MUV': np.arange(-21.5, -18, 0.5),
             'Ms': [9.61, 9.5, 9.21, 9.13, 8.96, 8.81, 8.75],
             'err': [0.39, 0.57, 0.47, 0.51, 0.56, 0.53, 0.57]},
         5: None,
         6: {'MUV': np.arange(-21.5, -18.5, 0.5),
             'Ms': [9.34, 9.23, 9.21, 9.14, 8.90, 8.77],
             'err': [0.44, 0.38, 0.41, 0.38, 0.38, 0.47]},
        }

        for j, z in enumerate(redshifts):
            if z not in data:
                continue
        
            ax_MsMUV.errorbar(data[z]['MUV'], data[z]['Ms'], yerr=data[z]['err'],
                color=colors[j], label='Salmon+ 2015' if j==0 else None, 
                fmt='o', mfc='none', **mkw)

        ax_MsMUV.legend(loc='upper right', fontsize=8)
                
    def _MegaPlotCleanup(self, kw):
        
        
        ax_sfe = kw['ax_sfe']
        ax_fco = kw['ax_fco']
        ax_rdu = kw['ax_rdu']
        ax_phi = kw['ax_phi']
        ax_bet = kw['ax_bet']
        
        
        ax_smf    = kw['ax_smf']
        ax_smhm   = kw['ax_smhm']
        ax_MsMUV  = kw['ax_MsMUV']
        ax_AUV    = kw['ax_AUV']
        ax_sfrd   = kw['ax_sfrd']
        ax_lae_z  = kw['ax_lae_z']
        ax_lae_m  = kw['ax_lae_m']
        ax_sfms   = kw['ax_sfms']
        
        ax_sfe.set_title('Model Inputs', fontsize=18)

        ax_rdu.set_xlabel(r'$M_h / M_{\odot}$')
        ax_sfe.set_ylabel(r'$f_{\ast} \equiv \dot{M}_{\ast} / f_b \dot{M}_h$')
        
        ax_fco.set_ylabel(r'$f_{\mathrm{cov,dust}}$')
            
        ax_rdu.set_ylabel(r'$R_{\mathrm{dust}} \ [\mathrm{kpc}]$')
        
        ax_AUV.set_title('Predictions', fontsize=18)
        ax_smf.set_title('Predictions', fontsize=18)
        ax_sfrd.set_title('Predictions', fontsize=18)

        ax_smf.set_ylabel(labels['galaxy_smf'])
        ax_smhm.set_xlabel(r'$M_h / M_{\odot}$')
        ax_smhm.set_ylabel(r'$M_{\ast} / M_h$')
        ax_phi.set_ylabel(labels['galaxy_lf'])
        ax_bet.set_ylabel(r'$\beta$')

        
        ax_MsMUV.set_ylabel(r'$\log_{10} M_{\ast} / M_{\odot}$')
        ax_MsMUV.set_xlabel(r'$M_{\mathrm{UV}}$')

        ax_AUV.set_xlabel(r'$M_{\mathrm{UV}}$')
        ax_AUV.set_ylabel(r'$A_{\mathrm{UV}}$')
        
        ax_sfms.set_xlabel(r'$M_{\ast} / M_{\odot}$')
        ax_sfms.set_ylabel(r'$\dot{M}_{\ast} \ [M_{\odot} \ \mathrm{yr}^{-1}]$')

        ax_sfrd.set_xlabel(r'$z$')
        ax_sfrd.set_ylabel(labels['sfrd'])
        ax_sfrd.set_ylim(1e-4, 1e-1)

        ax_lae_z.set_xlabel(r'$z$')
        ax_lae_z.set_ylabel(r'$X_{\mathrm{LAE}}, 1 - f_{\mathrm{cov}}$')
        ax_lae_m.set_xlabel(r'$M_{\mathrm{UV}}$')
        ax_lae_m.set_ylabel(r'$X_{\mathrm{LAE}}, 1 - f_{\mathrm{cov}}$')
        

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
        ax_sfe.legend(loc='lower right', fontsize=8, frameon=True, handlelength=1)
        ax_lae_z.legend(loc='upper left', frameon=True, fontsize=6)
        ax_lae_m.legend(loc='upper left', frameon=True, fontsize=6)
        ax_MsMUV.legend(loc='upper right', fontsize=8)
        
        
        self._MegaPlotLimitsAndTicks(kw)