"""

ModelFit.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Apr 28 11:19:03 MDT 2014

Description: For analysis of MCMC fitting.

"""

import numpy as np
import re, pickle, os
from ..util import labels
import matplotlib.pyplot as pl
from rt1d.physics.Constants import nu_0_mhz
from ..util.ReadData import read_pickled_chain
from ..util.SetDefaultParameterValues import SetAllDefaults

try:
    from multiplot import multipanel
except ImportError:
    pass

from ..util.Stats import Gauss1D, GaussND, error_1D, rebin

try:
    import emcee
except ImportError:
    pass

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1

def_inset_pars = \
{
 'size':None,
 'margin':(0,0), 
 'align':(0,1),
}

def_kwargs = \
{
 'labels': True,
}

suffixes = ['chain', 'pinfo', 'logL', 'blobs', 'binfo']

def parse_blobs(name):
    nsplit = name.split('_')
    
    if len(nsplit) == 2:
        pre, post = nsplit
    elif len(nsplit) == 3:
        pre, mid, post = nsplit
    
        pre = pre + mid
    
    if pre in labels:
        pass
        
    return None 
    
def logify_str(s):
    return r'$\mathrm{log}_{10}' + str(s.replace('$', '')) + '$'
    
def err_str(label, mu, err, log):
    l = str(label.replace('$', ''))
    
    if log:
        s = '\mathrm{log}_{10}' + l 
        
    else:
        s = l 
    
    s += '=%.3g^{+%.2g}_{-%.2g}' % (mu, err[0], err[1])
    
    return r'$%s$' % s
    
def def_par_names(N):
    return [i for i in np.arange(N)]

def def_par_labels(i):
    return 'parameter # %i' % i
    
class DummySampler:
    def __init__(self):
        pass        

class ModelFit(object):
    def __init__(self, data):
        """
        Parameters
        ----------
        data : instance, str
            Either an emcee.EnsembleSampler instance or the prefix for
            a bunch of files ending in .chain.pkl, .pinfo.pkl, etc.

        """
        
        # Read in data from emcee.EmsembleSampler object
        if isinstance(data, emcee.EnsembleSampler):
            self.chain = data.flatchain
            self.Nd = int(self.chain.shape[-1])
            self.parameters = def_par_names(self.Nd)
            self.is_log = [False] * self.Nd
            self.blobs = None
            self.blob_names = []
        # Read in data from file (assumed to be pickled)
        elif type(data) == str:
            prefix = data
            
            # Read MCMC chain
            self.chain, self.logL = \
                read_pickled_chain('%s.chain.pkl' % prefix, 
                    logL='%s.logL.pkl' % prefix)
            
            self.Nd = int(self.chain.shape[-1])
            
            # Read parameter names and info
            if os.path.exists('%s.pinfo.pkl' % prefix):
                f = open('%s.pinfo.pkl' % prefix, 'rb')
                self.parameters, self.is_log = pickle.load(f)
                f.close()
            
                if rank == 0:
                    print "Loaded %s.pinfo.pkl." % prefix
            else:
                self.parameters = range(self.Nd)
                self.is_log = [False] * self.Nd
            
            if os.path.exists('%s.blobs.pkl' % prefix):
                try:
                    blobs = read_pickled_chain('%s.blobs.pkl' \
                        % prefix)
                                    
                    mask = np.zeros_like(blobs)    
                    mask[np.argwhere(np.isinf(blobs))] = 1
                    
                    self.blobs = np.ma.masked_array(blobs, mask=mask)
                except:
                    if rank == 0:
                        print "WARNING: Error loading blobs."    
                    
                f = open('%s.binfo.pkl' % prefix, 'rb')
                self.blob_names, self.blob_redshifts = \
                    map(list, pickle.load(f))
                f.close()
                
                if rank == 0:
                    print "Loaded %s.binfo.pkl." % prefix
                
            else:
                self.blobs = self.blob_names = self.blob_redshifts = None
        else:
            raise TypeError('Argument must be emcee.EnsembleSampler instance or filename prefix')
                            
    @property
    def ref_pars(self):
        if not hasattr(self, '_ref_pars'):
            self._ref_pars = SetAllDefaults()
        
        return self._ref_pars
        
    def get_levels(self, L, nu=[0.99, 0.95, 0.68]):
        """
        Return levels corresponding to input nu-values, and assign
        colors to each element of the likelihood.
        """
    
        nu, levels = self.confidence_regions(L, nu=nu)
    
        tmp = L.ravel() / L.max()
                                                                      
        return nu, levels
    
    def get_1d_error(self, par, bins=20, nu=0.68):
        """
        Compute 1-D error bar for input parameter.
        
        Parameters
        ----------
        par : str
            Name of parameter. 
        bins : int
            Number of bins to use in histogram
        nu : float
            Percent likelihood enclosed by this 1-D error
        
        Returns
        -------
        Tuple, (maximum likelihood value, negative error, positive error).
        
        """
        
        j = self.parameters.index(par)
        
        hist, bin_edges = \
            np.histogram(self.chain[:,j], density=True, bins=bins)

        bc = rebin(bin_edges)

        mu, sigma = float(bc[hist == hist.max()]), error_1D(bc, hist, nu=nu)   
        
        return mu, np.array(sigma)
        
    def _get_1d_kwargs(self, **kw):
        
        for key in ['labels', 'colors', 'linestyles']:
        
            if key in kw:
                kw.pop(key)

        return kw
                
    def posterior_pdf(self, pars, z=None, ax=None, fig=1, multiplier=[1.]*2,
        nu=[0.99, 0.95, 0.68], slc=None, overplot_nu=False, density=True, 
        color_by_nu=False, contour=True, filled=True, take_log=[False]*2,
        bins=20, xscale='linear', yscale='linear', skip=0, skim=1, **kwargs):
        """
        Compute posterior PDF for supplied parameters. 
    
        If len(pars) == 2, plot 2-D posterior PDFs. If len(pars) == 1, plot
        1-D marginalized PDF.
    
        Parameters
        ----------
        pars : str, list
            Name of parameter or list of parameters to analyze.
        z : float
            Redshift, if any element of pars is a "blob" quantity.
        plot : bool
            Plot PDF?
        nu : float, list
            If plot == False, return the nu-sigma error-bar.
            If color_by_nu == True, list of confidence contours to plot.
        color_by_nu : bool
            If True, color points based on what confidence contour they lie
            within.
        contour : bool
            Use contours rather than discrete points?
        multiplier : list
            Two-element list of multiplicative factors to apply to elements of
            pars.
        take_log : list
            Two-element list saying whether to histogram the base-10 log of
            each parameter or not.
        skip : int
            Number of steps at beginning of chain to exclude. This is a nice
            way of doing a burn-in after the fact.
        skim : int
            Only take every skim'th step from the chain.

        Returns
        -------
        Either a matplotlib.Axes.axis object or a nu-sigma error-bar, 
        depending on whether we're doing a 2-D posterior PDF (former) or
        1-D marginalized posterior PDF (latter).
    
        """
    
        kw = def_kwargs.copy()
        kw.update(kwargs)
        
        if type(pars) != list:
            pars = [pars]
        if type(take_log) == bool:
            take_log = [take_log] * len(pars)
        if type(multiplier) == bool:
            multiplier = [multiplier] * len(pars)            
    
        if ax is None:
            gotax = False
            fig = pl.figure(fig)
            ax = fig.add_subplot(111)
        else:
            gotax = True
    
        if type(pars) not in [list, tuple]:
            pars = [pars]

        binvec = []
        to_hist = []        
        is_log = []
        for k, par in enumerate(pars):

            if par in self.parameters:        
                j = self.parameters.index(par)
                is_log.append(self.is_log[j])
                
                val = self.chain[skip:,j].ravel()[::skim] * multiplier[k]
                if take_log[k]:
                    to_hist.append(np.log10(val))
                else:
                    to_hist.append(val)
                
            elif par in self.blob_names:
                
                if z is None:
                    raise ValueError('Must supply redshift!')
                    
                i = self.blob_redshifts.index(z)
                j = list(self.blob_names).index(par)
                
                is_log.append(False)
                
                val = self.blobs[skip:,i,j].compressed()[::skim] * multiplier[k]
                if take_log[k]:
                    to_hist.append(np.log10(val))
                else:
                    to_hist.append(val)
                
            else:
                raise ValueError('Unrecognized parameter %s' % str(par))

            if type(bins) == int:
                binvec.append(bins)
            else:
                binvec.append(bins[k])

        if len(pars) == 1:

            hist, bin_edges = \
                np.histogram(to_hist[0], density=density, 
                    bins=bins)

            bc = rebin(bin_edges)
        
            tmp = self._get_1d_kwargs(**kw)

            ax.plot(bc, hist / hist.max(), drawstyle='steps-mid', **tmp)
            ax.set_xscale(xscale)
            ax.set_ylim(0, 1.05)
            
            if overplot_nu:
                
                try:
                    mu, sigma = bc[hist == hist.max()], error_1D(bc, hist, nu=nu)
                except ValueError:
                    mu, sigma = bc[hist == hist.max()], error_1D(bc, hist, nu=nu[0])
                
                mi, ma = ax.get_ylim()
            
                ax.plot([mu - sigma[0]]*2, [mi, ma], color='k', ls=':')
                ax.plot([mu + sigma[1]]*2, [mi, ma], color='k', ls=':')
            
        else:
    
            # Compute 2-D histogram
            hist, xedges, yedges = \
                np.histogram2d(to_hist[0], to_hist[1], 
                    bins=[binvec[0], binvec[1]])

            hist = hist.T

            # Recover bin centers
            bc = []
            for i, edges in enumerate([xedges, yedges]):
                bc.append(rebin(edges))
                    
            # Determine mapping between likelihood and confidence contours
            if color_by_nu:
    
                # Get likelihood contours (relative to peak) that enclose
                # nu-% of the area
                nu, levels = self.get_levels(hist, nu=nu)
    
                if filled:
                    ax.contourf(bc[0], bc[1], hist / hist.max(), 
                        levels, **kwargs)
                else:
                    ax.contour(bc[0], bc[1], hist / hist.max(),
                        levels, **kwargs)
                
            else:
                if filled:
                    ax.contourf(bc[0], bc[1], hist / hist.max(), **kw)
                else:
                    ax.contour(bc[0], bc[1], hist / hist.max(), **kw)

            ax.set_xscale(xscale)
            ax.set_yscale(yscale)
            
            if overplot_nu:
                
                for i in range(2):
                    
                    hist, bin_edges = \
                        np.histogram(to_hist[i], density=density, bins=bins)
                    
                    bc = rebin(bin_edges)
                    
                    mu = bc[hist == hist.max()]
                    
                    try:
                        sigma = error_1D(bc, hist, nu=nu)
                    except ValueError:
                        
                        sigma = error_1D(bc, hist, nu=nu[0])
                    
                    if i == 0:
                        mi, ma = ax.get_ylim()
                    else:
                        mi, ma = ax.get_xlim()
                                        
                    if i == 0:
                        ax.plot([mu - sigma[0]]*2, [mi, ma], color='k', ls=':')
                        ax.plot([mu + sigma[1]]*2, [mi, ma], color='k', ls=':')
                    else:
                        ax.plot([mi, ma], [mu - sigma[0]]*2, color='k', ls=':')
                        ax.plot([mi, ma], [mu + sigma[1]]*2, color='k', ls=':')

        if kw['labels']:
            self.set_axis_labels(ax, pars, is_log, take_log)

        pl.draw()

        return ax
        
    def extract_blob(self, name, z):
        i = self.blob_redshifts.index(z)
        j = list(self.blob_names).index(name)
        
        return self.blobs[:,i,j].compressed()
        
    def max_likelihood_parameters(self):
        """
        Return parameter values at maximum likelihood point.
        """
        
        iML = np.argmax(self.logL)
        
        p = {}
        for i, par in enumerate(self.parameters):
            if self.is_log[i]:
                p[par] = 10**self.chain[iML,i]
            else:
                p[par] = self.chain[iML,i]
            
        return p
        
    def triangle_plot(self, pars, z=None, panel_size=(0.5,0.5), padding=(0,0),
        show_errors=False, take_log=False, multiplier=1,
        fig=1, plot_inputs=False, inputs={}, tighten_up=0.0, 
        inset_data=None, bins=20, mp=None, skip=0, skim=1,
        filled=True, inset=False, inset_pars={}, **kwargs):
        """
        Make an NxN panel plot showing 1-D and 2-D posterior PDFs.
        
        Parameters
        ----------
        pars : list
            Parameters to include in triangle plot.
            1-D PDFs along diagonal will follow provided order of parameters
            from left to right.
        fig : int
            ID number for plot window. 
        bins : int,
            Number of bins in each dimension.
        skip : int
            Number of steps at beginning of chain to exclude. This is a nice
            way of doing a burn-in after the fact.
        skim : int
            Only take every skim'th step from the chain.
        
        Returns
        -------
        multiplot.multipanel instance.
        
        If inset=True, returns multiplot.multipanel instance in addition to
        an axes object representing the inset.
            
        """    
        
        kw = def_kwargs.copy()
        kw.update(kwargs)
        
        if type(take_log) == bool:
            take_log = [take_log] * len(pars)
        if multiplier == 1:
            multiplier = [multiplier] * len(pars)        
        if type(bins) == int:
            bins = [bins] * len(pars)
        
        is_log = []
        for par in pars[-1::-1]:
            if par in self.parameters:
                is_log.append(self.is_log[self.parameters.index(par)])
            elif par in self.blob_names:
                is_log.append(False)

        Nd = len(pars)
                           
        # Multipanel instance
        had_mp = True
        if mp is None:
            had_mp = False
            mp = multipanel(dims=[Nd]*2, padding=padding, diagonal='lower',
                panel_size=panel_size, num=fig, **kw)

        # Loop over parameters
        for i, p1 in enumerate(pars[-1::-1]):
            for j, p2 in enumerate(pars):

                # Row number is i
                # Column number is self.Nd-j-1

                k = mp.axis_number(i, j)

                if k is None:
                    continue
                
                if mp.grid[k] is None:
                    continue

                # Input values (optional)
                if p1 in self.ref_pars or inputs:
                    if not inputs:
                        val = self.ref_pars[p1]
                    else:
                        val = inputs[p1]
                        
                    if is_log[i]:
                        yin = np.log10(val)    
                    else:
                        yin = val                        
                else:
                    yin = None
                
                if p2 in self.ref_pars or inputs:     
                    if not inputs:
                        val = self.ref_pars[p2]
                    else:
                        val = inputs[p2]
                        
                    if is_log[Nd-j-1]:
                        xin = np.log10(val)
                    else:
                        xin = val

                else:
                    xin = None

                col, row = mp.axis_position(k)    
                                        
                # 1-D PDFs on the diagonal    
                if k in mp.diag:                    
                    self.posterior_pdf(p1, ax=mp.grid[k], 
                        take_log=take_log[-1::-1][i], z=z,
                        multiplier=[multiplier[-1::-1][i]], 
                        bins=bins[-1::-1][i], skip=skip, skim=skim, **kw)
                    
                    if col != 0:
                        mp.grid[k].set_ylabel('')
                    if row != 0:
                        mp.grid[k].set_xlabel('')    
                    
                    if show_errors:
                        mu, err = self.get_1d_error(p1)
                                                 
                        mp.grid[k].set_title(err_str(labels[p1], mu, err, 
                            self.is_log[i])) 
                     
                    if not plot_inputs:
                        continue
                        
                    if xin is not None:
                        mp.grid[k].plot([xin]*2, [0, 1.05], 
                            color='k', ls=':', lw=2)    
                            
                    continue
                                        
                # 2-D PDFs elsewhere
                self.posterior_pdf([p2, p1], ax=mp.grid[k], z=z,
                    take_log=[take_log[j], take_log[-1::-1][i]],
                    multiplier=[multiplier[j], multiplier[-1::-1][i]], 
                    bins=[bins[j], bins[-1::-1][i]], filled=filled, **kw)
                
                if row != 0:
                    mp.grid[k].set_xlabel('')
                if col != 0:
                    mp.grid[k].set_ylabel('')

                # Input values
                if not plot_inputs:
                    continue

                if xin is not None:
                    mp.grid[k].plot([xin]*2, mp.grid[k].get_ylim(), color='k', 
                        ls=':')
                if yin is not None:
                    mp.grid[k].plot(mp.grid[k].get_xlim(), [yin]*2, color='k', 
                        ls=':')

        mp.grid[np.intersect1d(mp.left, mp.top)[0]].set_yticklabels([])
        
        mp.fix_ticks()
        mp.rescale_axes(tighten_up=tighten_up)
                
        if not inset:
            return mp
        
        if not inset_pars:
            inset_pars = def_inset_pars
        else:
            tmp = def_inset_pars.copy()
            tmp.update(inset_pars)
            inset_pars = tmp
            
        l = mp.window['left'] + (inset_pars['align'][0] + 1) \
            * mp.window['pane'][0] + inset_pars['margin'][0]

        b = mp.window['bottom'] + (inset_pars['align'][1] + 1) \
            * mp.window['pane'][1] + inset_pars['margin'][1]
        
        if inset_pars['size'] is None:
            w = mp.window['pane'][0] \
                + (1. - mp.window['right']) - mp.window['left']
            h = mp.window['pane'][1] \
                + (1. - mp.window['top']) - mp.window['bottom']
        else:
            w, h = inset_pars['size']
        
        # Draw signals from distribution
        inset = mp.fig.add_axes([l, b, w, h])

        pl.draw()
    
        return mp, inset
        
    def set_axis_labels(self, ax, pars, is_log, take_log=False):
        """
        Make nice axis labels.
        """
    
        if type(take_log) == bool:
            take_log = [take_log] * len(pars)
    
        if pars[0] in labels:
            if is_log[0] or take_log[0]:
                ax.set_xlabel(logify_str(labels[pars[0]]))
            else:
                ax.set_xlabel(labels[pars[0]])
        elif type(pars[0]) == int:
            ax.set_xlabel(def_par_labels(pars[0]))
        else:
            ax.set_xlabel(pars[0])
    
        if len(pars) == 1:
            ax.set_ylabel('PDF')
    
            pl.draw()
            return
    
        if pars[1] in labels:
            if is_log[1] or take_log[1]:
                ax.set_ylabel(logify_str(labels[pars[1]]))
            else:
                ax.set_ylabel(labels[pars[1]])
        elif type(pars[1]) == int:
            ax.set_ylabel(def_par_labels(pars[1]))
        else:
            ax.set_ylabel(pars[1])
    
        pl.draw()        
        
    def confidence_regions(self, L, nu=[0.99, 0.95, 0.68]):
        """
        Integrate outward at "constant water level" to determine proper
        2-D marginalized confidence regions.
    
        Note: this is fairly crude.
    
        Parameters
        ----------
        L : np.ndarray
            Grid of likelihoods.
        nu : float, list
            Confidence intervals of interest.
    
        Returns
        -------
        List of contour values (relative to maximum likelihood) corresponding 
        to the confidence region bounds specified in the "nu" parameter, 
        in order of decreasing nu.
        """
    
        if type(nu) in [int, float]:
            nu = np.array([nu])
    
        # Put nu-values in ascending order
        if not np.all(np.diff(nu) > 0):
            nu = nu[-1::-1]
    
        peak = float(L.max())
        tot = float(L.sum())
    
        # Counts per bin in descending order
        Ldesc = np.sort(L.ravel())[-1::-1]
    
        j = 0  # corresponds to whatever contour we're on
    
        Lprev = 1.0
        Lencl_prev = 0.0
        contours = [1.0]
        for i in range(1, Ldesc.size):
    
            # How much area (fractional) is enclosed within the current contour?
            Lencl_now = L[L >= Ldesc[i]].sum() / tot
    
            Lnow = Ldesc[i]
    
            # Haven't hit next contour yet
            if Lencl_now < nu[j]:
                pass
    
            # Just passed a contour
            else:
                # Interpolate to find contour more precisely
                Linterp = np.interp(nu[j], [Lencl_prev, Lencl_now],
                    [Ldesc[i-1], Ldesc[i]])
                # Save relative to peak
                contours.append(Linterp / peak)
    
                j += 1
    
            Lprev = Lnow
            Lencl_prev = Lencl_now
    
            if j == len(nu):
                break
    
        # Return values that match up to inputs    
        return nu[-1::-1], contours[-1::-1]
    
    def errors_to_latex(self, pars, nu=0.68, in_units=None, out_units=None):
        """
        Output maximum-likelihood values and nu-sigma errors ~nicely.
        """
                
        if type(nu) != list:
            nu = [nu]
            
        hdr = 'parameter    '
        for conf in nu:
            hdr += '%.1f' % (conf * 100)
            hdr += '%    '
        
        print hdr
        print '-' * len(hdr)    
        
        for i, par in enumerate(pars):
            
            s = str(par)
            
            for j, conf in enumerate(nu):
                
                
                mu, sigma = \
                    map(np.array, self.get_1d_error(par, bins=100, nu=conf))

                if in_units and out_units != None:
                    mu, sigma = self.convert_units(mu, sigma,
                        in_units=in_units, out_units=out_units)

                s += r" & $%5.3g_{-%5.3g}^{+%5.3g}$   " % (mu, sigma[0], sigma[1])
        
            s += '\\\\'
            
            print s
    
    def convert_units(self, mu, sigma, in_units, out_units):
        """
        Convert units on common parameters of interest.
        
        So far, just equipped to handle frequency -> redshift and Kelvin
        to milli-Kelvin conversions. 
        
        Parameters
        ----------
        mu : float
            Maximum likelihood value of some parameter.
        sigma : np.ndarray
            Two-element array containing asymmetric error bar.
        in_units : str
            Units of input mu and sigma values.
        out_units : str
            Desired units for output.
        
        Options
        -------
        in_units and out_units can be one of :
        
            MHz
            redshift
            K
            mK
            
        Returns
        -------
        Tuple, (mu, sigma). Remember that sigma is itself a two-element array.
            
        """
        
        if in_units == 'MHz' and out_units == 'redshift':
            new_mu = nu_0_mhz / mu - 1.
            new_sigma = abs(new_mu - (nu_0_mhz / (mu + sigma[1]) - 1.)), \
                abs(new_mu - (nu_0_mhz / (mu - sigma[0]) - 1.))
                        
        elif in_units == 'redshift' and out_units == 'MHz':
            new_mu = nu_0_mhz / (1. + mu)
            new_sigma = abs(new_mu - (nu_0_mhz / (1. + mu - sigma[0]))), \
                        abs(new_mu - (nu_0_mhz / (1. + mu - sigma[1])))
        elif in_units == 'K' and out_units == 'mK':
            new_mu = mu * 1e3
            new_sigma = np.array(sigma) * 1e3
        elif in_units == 'mK' and out_units == 'K':
            new_mu = mu / 1e3
            new_sigma = np.array(sigma) / 1e3
        else:
            raise ValueError('Unrecognized unit combination')
        
        return new_mu, new_sigma
    