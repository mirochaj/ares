"""

ObservedLF.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Thu Jan 28 12:38:11 PST 2016

Description: 

"""

import numpy as np
from ..util import read_lit
import matplotlib.pyplot as pl
from .MultiPlot import MultiPanel
from matplotlib.patches import Patch

datasets_lf = ('oesch2013', 'oesch2014', 'bouwens2015', 'atek2015', 
    'parsa2016', 'finkelstein2015', 'vanderburg2010')
datasets_smf = ('song2016', )

colors = ['m', 'c', 'r', 'y', 'g', 'b'] * 3
markers = ['o'] * 6 + ['s'] * 6    
    
default_colors = {}
default_markers = {}    
for i, dataset in enumerate(datasets_lf):
    default_colors[dataset] = colors[i]
    default_markers[dataset] = markers[i]

for i, dataset in enumerate(datasets_smf):
    default_colors[dataset] = colors[i]
    default_markers[dataset] = markers[i]

_ulim_tick = 0.5

class GalaxyPopulation(object):
    def __init__(self):
        pass

    def compile_data(self, redshift, sources='all', round_z=False,
        quantity='lf'):
        """
        Create a master dictionary containing the MUV points, phi points,
        and (possibly asymmetric) errorbars for all (or some) data available.
        
        Parameters
        ----------
        z : int, float
            Redshift, dummy!

        """

        data = {}

        if sources == 'all':
            if quantity == 'lf':
                sources = datasets_lf
            else:
                sources = datasets_smf
        elif type(sources) is str:
            sources = [sources]    

        for source in sources:

            src = read_lit(source)

            if redshift not in src.redshifts and (not round_z):
                print "No z=%g data in %s" % (redshift, source)
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
            data[source]['wavelength'] = src.wavelength
            
            M = src.data[quantity][z]['M']            
            if hasattr(M, 'data'):
                data[source]['M'] = M.data
            else:
                data[source]['M'] = np.array(M)
            
            if src.units['phi'] == 'log10':
                err_lo = []; err_hi = []; uplims = []
                for i, err in enumerate(src.data[quantity][z]['err']):
                    
                    if type(err) not in [int, float]:
                        err = np.mean(err)
                        #raise NotImplemented('help!')
                        print "WARNING: log10 asymmetric errors not great!"
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
                        err_hi.append(err[1])
                        err_lo.append(err[0])
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
            AUV=None, wavelength=1600., sed_model=None, **kwargs):
                
        return self.Plot(z=z, ax=ax, fig=fig, sources=sources, round_z=round_z,
            AUV=AUV, wavelength=1600, sed_model=None, quantity='lf', **kwargs)  
        
    def PlotSMF(self, z, ax=None, fig=1, sources='all', round_z=False, 
            AUV=None, wavelength=1600., sed_model=None, **kwargs):
    
        return self.Plot(z=z, ax=ax, fig=fig, sources=sources, round_z=round_z,
            AUV=AUV, wavelength=1600, sed_model=None, quantity='smf', **kwargs)              
                
    def Plot(self, z, ax=None, fig=1, sources='all', round_z=False, 
        AUV=None, wavelength=1600., sed_model=None, quantity='lf', **kwargs):
        """
        Plot the luminosity function data at a given redshift.
        
        Parameters
        ----------
        z : int, float
            Redshift of interest
        wavelength : int, float 
            Wavelength (in Angstroms) of LF. 
        sed_model : instance
            ares.populations.SynthesisModel
            
        """
        
        if ax is None:
            gotax = False
            fig = pl.figure(fig)
            ax = fig.add_subplot(111)
        else:
            gotax = True
            
        data = self.compile_data(z, sources, round_z=round_z, quantity=quantity)
        
        if sources == 'all':
            if quantity == 'lf':
                sources = datasets_lf
            else:
                sources = datasets_smf
        elif type(sources) is str:
            sources = [sources]
            
        for source in sources:
            if source not in data:
                continue
                                        
            M = np.array(data[source]['M'])
            phi = np.array(data[source]['phi'])
            err = np.array(data[source]['err'])
            ulim = np.array(data[source]['ulim'])
                                                
            if not kwargs:
                kw = {'fmt':'o', 'ms':5, 'elinewidth':2, 
                    'mec':default_colors[source], 
                    'fmt': default_markers[source],
                    'color':default_colors[source], 'capthick':2}
            else:
                kw = kwargs
            
            if AUV is not None:
                dc = AUV(z, np.array(M))
            else:
                dc = 0
                
            # Shift band [optional]
            if data[source]['wavelength'] != wavelength:
                #shift = sed_model.
                print "WARNING: %s wavelength=%iA, not %iA!" \
                    % (source, data[source]['wavelength'], wavelength)
            #else:
            shift = 0.    
              
            ax.errorbar(M+shift-dc, phi, yerr=err, uplims=ulim, zorder=10, **kw)
        
        ax.set_yscale('log', nonposy='clip')    
        
        if quantity == 'lf':
            ax.set_xlim(-26.5, -10)
            ax.set_xticks(np.arange(-26, -10, 1), minor=True)
            ax.set_xlabel(r'$M_{\mathrm{UV}}$')    
            ax.set_ylabel(r'$\phi(M_{\mathrm{UV}}) \ [\mathrm{mag}^{-1} \ \mathrm{cMpc}^{-3}]$')
        elif quantity == 'smf':
            ax.set_xscale('log')
            ax.set_xlim(1e7, 1e13)
            ax.set_xlabel(r'$M_{\ast} / M_{\odot}$')    
            ax.set_ylabel(r'$\phi(M_{\ast}) \ [\mathrm{dex}^{-1} \ \mathrm{cMpc}^{-3}]$')
            
        
        pl.draw()
        
        return ax
            
    def MultiPlot(self, redshifts, sources='all', round_z=False, ncols=1, 
        panel_size=(0.75,0.75), fig=1, ymax=10, legends=None, AUV=None,
        quantity='lf'):
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
            nrows = len(redshifts) / ncols
            
        if nrows * ncols != len(redshifts):
            nrows += 1
            
        dims = (nrows, ncols)    
            
        # Force redshifts to be in ascending order
        if not np.all(np.diff(redshifts)) > 0:   
            redshifts = np.sort(redshifts)
            
        # Create multiplot
        mp = MultiPanel(dims=dims, panel_size=panel_size, fig=fig, 
            padding=[0.2]*2)
        
        self.redshifts_in_mp = []
        for i, z in enumerate(redshifts):
            k = mp.elements.ravel()[i]
            ax = mp.grid[k]
            
            # Where in the MultiPlot grid are we?
            self.redshifts_in_mp.append(k)
                        
            self.Plot(z, sources=sources, round_z=round_z, ax=ax, AUV=AUV,
                quantity=quantity)
            
            ax.annotate(r'$z \sim %i$' % (round(z)), (0.05, 0.95), 
                ha='left', va='top', xycoords='axes fraction')
        
        mp.fix_ticks(rotate_x=45)
                
        for i, z in enumerate(redshifts):
            k = mp.elements.ravel()[i]
            ax = mp.grid[k]
            
            if quantity == 'lf':
                ax.set_xlim(-24, -14.)
                ax.set_ylim(1e-7, 1e-1)
                ax.set_xticks(np.arange(-23, -13, 2), minor=True)
                ax.set_yscale('log', nonposy='clip')  
                ax.set_ylabel('')
            else:
                ax.set_xscale('log')
                ax.set_xlim(1e6, 1e12)
                ax.set_ylim(1e-9, 15)
                ax.set_yscale('log', nonposy='clip')                      
            
        if quantity == 'lf':
            mp.global_ylabel(r'$\phi(M_{\mathrm{UV}}) \ [\mathrm{mag}^{-1} \ \mathrm{cMpc}^{-3}]$')
        else:
            mp.global_ylabel(r'$\phi(M_{\ast}) \ [\mathrm{dex}^{-1} \ \mathrm{cMpc}^{-3}]$')
            
        return mp
            
    def annotated_legend(self, ax, loc=(0.95, 0.05), sources='all'):   
        """
        Annotate sources properly color-coded.
        """     
        if sources == 'all':
            sources = all_datasets
                    
        for i, source in enumerate(sources):
            coord = (loc[0], loc[1] + 0.05 * i)    
            ax.annotate(source, coord, fontsize=14, 
                color=default_colors[source], ha='right', va='bottom',
                xycoords='axes fraction')
        
        pl.draw()
        
        return ax
            