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
from ..phenom import DustCorrection
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
from ..physics.Constants import rhodot_cgs
from .MultiPlot import MultiPanel, add_master_legend
from ..util.Stats import symmetrize_errors, bin_samples
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
        **kwargs):
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
            
            if use_best:
                bkw = pop.base_kwargs.copy()
                bkw.update(pop.max_likelihood_parameters(method=method))
                bkw['conserve_memory'] = False
                
                pop = GalaxyEnsemble(**bkw)
                self._MegaPlotPop(axes, pop)
            else:
                self._MegaPlotChain(axes, pop, **kwargs)
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
            phi = pop.LuminosityFunction(z, _mags, batch=True)
            ax_phi.semilogy(_mags, phi, color=colors[j], drawstyle='steps-mid')
                    
            # Binned version
            _mags_b, _beta, _std = pop.Beta(z, wave=1600, return_binned=True,
                Mbins=np.arange(-25, -10, 1.0), batch=True)
            
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
            
            mags, beta, std = pop.Beta(z, wave=1600., return_binned=False, 
                batch=True)
            
            # MUV-Mstell
            _x, _y, _z = bin_samples(mags, np.log10(Ms), _mags_b)
            ax_MsMUV.plot(_x, _y, color=colors[j])    
            
            # Beta just to get 'mags'
            if pop.pf['pop_dust_yield'] == 0:
                xa_f.append(0)
                xa_b.append(0)
                
                ax_bet.plot(_mags_b, dc1.Beta(z, _mags_b), color=colors[j])
                
                continue
                
            ax_bet.plot(_mags_b, _beta, color=colors[j])    
            
            fcov = pop.guide.dust_fcov(z=z, Mh=Mh)
            Rdust = pop.guide.dust_scale(z=z, Mh=Mh)
            
            if type(fcov) in [int, float, np.float64]:
                fcov = fcov * np.ones_like(Mh)
            
            #any_fcov = np.any(np.diff(fcov, axis=1) != 0)
            #any_fduty = np.any(np.diff(fduty, axis=1) != 0)
            
            if fcov.ndim == 1:
                try:
                    fduty = pop.guide.fduty(z=z, Mh=Mh)
                    ax_fco.semilogx(Mh, fduty, color=colors[j])
                except:
                    pass    
            else:  
                ax_fco.semilogx(Mh, fcov, color=colors[j])
                
                
            ax_rdu.loglog(Mh, Rdust, color=colors[j])
                
            _mags_A, AUV, std = pop.AUV(z, wave=1600., return_binned=True,
                Mbins=np.arange(-25, -10, 1.))
            
            ax_AUV.plot(_mags_A, AUV, color=colors[j])
                            
            # LAE stuff
            _x, _y, _z = bin_samples(mags, fcov, _mags_A)
            ax_lae_m.plot(_x, 1. - _y, color=colors[j])
            
            faint  = np.logical_and(_mags_A >= -20.25, _mags_A < -18.)
            bright = _mags_A < -20.25
            
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
        #anl.DeriveBlob(name='x_LAE', expr='1-fcov', clobber=True,
        #    varmap={'fcov': 'dust_fcov'}, ivar=anl.get_ivars('dust_fcov'))    
                
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
            
            anl.ReconstructedFunction('beta_hst', ivar=[z, None], ax=ax_bet,
                color=colors[j], **kwargs)
            
            anl.ReconstructedFunction('AUV', ivar=[z, None], ax=ax_AUV,
                color=colors[j], **kwargs)
                
            anl.ReconstructedFunction('sfrd', ivar=None, ax=ax_sfrd,
                color=colors[j], **kwargs)
        
            anl.ReconstructedFunction('dust_scale', ivar=[z, None], ax=ax_rdu,
                color=colors[j], **kwargs)
            
            if 'fduty' in anl.all_blob_names:
                anl.ReconstructedFunction('fduty', ivar=[z, None], ax=ax_fco,
                    color=colors[j], **kwargs)
            else:    
                anl.ReconstructedFunction('dust_fcov', ivar=[z, None], ax=ax_fco,
                    color=colors[j], **kwargs)    
            
            #anl.ReconstructedFunction('x_LAE', ivar=[z, None], ax=ax_lae_m,
            #    color=colors[j], **kwargs)
                
                    
                
                
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