"""

SignalExtraction.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Aug 19 16:16:21 MDT 2013

Description: 

"""

import numpy as np
import matplotlib.pyplot as pl
import os, re, itertools
import matplotlib as mpl
from ..simple import Interpret21cm
from scipy.optimize import fsolve, fmin
from scipy.integrate import quad, dblquad
from rt1d.physics.Constants import nu_0_mhz

try:
    from jmpy.misc import readers
except ImportError:
    pass

try:
    from mathutils.stats import rebin, error_1D, GaussND
except ImportError:
    pass

try:
    from multiplot import multipanel
except ImportError:
    pass

defpars = {
    'ionosphere':False, 
    'sun':False,
    'moon':False, 
    'regions':2, 
    'integration':1000, 
    'fixTd':True,
    'instrument':'tight',
}

default_realization = \
{
 'zB': 30.,
 'TB': -5.,
 'zC': 20.,
 'TC': -100.,
}

default_errors = \
{
 'zB': 0.1,
 'TB': 5.,
 'Bcorr' : 0.,
 'zC': 0.1,
 'TC': 10.,
 'Ccorr': 0.,
}

GLORB = os.environ.get('GLORB')

# Not really columns in the files but you get the idea
cols = {'nu_B': 1, 'T_B': 2, 'nu_C': 3, 'T_C': 4, 'nu_D': 5}

class Extracted21cm(object):
    """ """
    def __init__(self, signal=None, model_grid=None, **kwargs):
        self.pf = defpars.copy()
        self.pf.update(kwargs)
        
        self.mg = model_grid    
            
        self.simpl = Interpret21cm(signal=signal)  
        self.cosm = self.simpl.cosm
        
        print 'WARNING: Need to merge Extracted21cm and ExtractedSignal.'
        
    @property
    def real21(self):
        if not hasattr(self, '_real21'):  
            self.set_realization()  
        
        return self._real21
        
    def set_realization(self, **kwargs):
        """
        Set true positions of turning points.
        
        Parameters
        ----------
            zB, TB, zC, TC, zD, TD
        """
        self._real21 = default_realization.copy() 
        self._real21.update(kwargs)   
        
    @property
    def real21err(self):
        if not hasattr(self, '_real21err'):
            self.set_errors()
    
        return self._real21err
    
    def set_errors(self, fn=None, **kwargs):
        """
        Set parameterized confidence regions. Assumes 2D Gaussian.
        
        Parameters
        ----------
            zB, TB, Bcorr, zC, TC, Ccorr, zD, TD, Dcorr
        """      
        
        if fn is None:
            self._real21err = default_errors.copy() 
            self._real21err.update(kwargs)
        else:
            raise NotImplemented('Need to re-implement errors from file option.')
        
    def str(self, pt):
        zp = 'z%s' % pt
        Tp = 'T%s' % pt
        cp = '%scorr' % pt
        
        return zp, Tp, cp
        
    #@property
    #def cov(self):
    #    if not hasattr(self, '_cov'):
    #        cov = np.array([[self.real21err[zp]**2, self.real21err[cp]],
    #                        [self.real21err[cp], self.real21err[Tp]**2]])
            
        
    def confidence(self, z, dTb, pt='B'):
        """
        Return confidence
        """    
        
        zp, Tp, cp = self.str(pt)
        
        cov = np.array([[self.real21err[zp]**2, self.real21err[cp]],
                        [self.real21err[cp], self.real21err[Tp]**2]])
                        
        mu = np.array([self.real21[zp], self.real21[Tp]])
        return GaussND(np.array([z, dTb]), mu, cov)
            
    
    def error_nu_sigma(self, zB=30., err=0.01, nu=0.675):        
        gauss = lambda x, mu, sigma: np.exp(-(x - mu)**2. / 2. / sigma**2.)
        
        zpdf = lambda zz: gauss(zz, zB, err*zB)
        area = quad(zpdf, zB-20, zB+20)[0]
        to_min = lambda x: (quad(zpdf, zB-x, zB+x)[0] / area) - nu

        zerr_nu_sig = fsolve(lambda x: to_min(x), err)
    
        return abs(zerr_nu_sig)
            
    def confidence_grid(self, zpts=100, Tpts=100, zlim=(10, 40), Tlim=(-50, 0),
        pt='B'):
        """
        Compute uncertainty of sigal extraction on a grid of points.

        Parameters
        ----------
        zpts : int
            Number of points in redshift space
        Tpts : int
            Number of points in differential brightness temperature space
        zlim : tuple
            (minimum redshift, maximum redshift) over which to sample
        Tlim : tuple
            (minimum temp, maximum temp) over which to sample
        
        Returns
        -------
        Do not need to transpose prior to contour plot-making!
        """
        
        z = np.linspace(zlim[0], zlim[1], zpts)
        T = np.linspace(Tlim[0], Tlim[1], Tpts)

        zz, TT = np.meshgrid(z, T)
        
        conf = np.zeros_like(zz)
        for combo in itertools.product(*(np.arange(zz.shape[0]), 
            np.arange(zz.shape[0]))):
            conf[combo] = self.confidence(zz[combo], TT[combo], pt=pt)
    
        # Normalize to integrated area
        conf /= conf.max()
            
        return z, T, conf
        
    def _find_sigma_contours(self, pt='B', maxiter=1e3):
        """
        Compute likelihood corresponding to 68% and 95% confidence interval.
        """
        
        print "Computing confidence intervals..."
        
        zp, Tp, cp = self.str(pt)
        
        mu_z = self.real21[zp]
        mu_T = self.real21[Tp]
        dz = self.real21err[zp]
        dT = self.real21err[Tp]
        
        cov = np.array([[self.real21err[zp]**2, self.real21err[cp]],
                        [self.real21err[cp], self.real21err[Tp]**2]])
        
        norm = self.confidence(mu_z, mu_T, pt=pt)
        Atot = np.sqrt((2. * np.pi)**2. / np.linalg.det(cov)) / 4.    
            
        i = 0
        zstep = dz / 10. 
        area_last = 0.0
        area_now = 0.0
        levels = []
        while True:
            
            # Pick a delta z, find corresponding delta T contour that
            # keeps the likelihood constant
            
            # Contour value we're looking for
            ref = self.confidence(mu_z + zstep, mu_T, pt=pt)
                        
            get_minor = lambda DT: \
                self.confidence(mu_z, mu_T + DT, pt=pt) - ref
            
            Tstep = abs(fsolve(get_minor, 0.1)[0])
            
            T_of_z = lambda z: mu_T + Tstep * np.sqrt(1. - (z-mu_z)**2 / zstep**2)
            
            # Compute area within this contour
            area_now = 4. \
                * abs(dblquad(lambda y, x: self.confidence(x, y, pt=pt) / norm, 
                mu_z-0.99999999*zstep, mu_z,
                lambda z: T_of_z(z), lambda z: mu_T)[0]) / Atot
                               
            if area_last < 0.68 and area_now > 0.68:
                levels.append(self.confidence(mu_z - zstep, mu_T - Tstep, 
                    pt=pt) / norm)    
                
            if area_last < 0.95 and area_now > 0.95:
                levels.append(self.confidence(mu_z - zstep, mu_T - Tstep, 
                    pt=pt) / norm)
            
            
            area_last = area_now
            zstep += dz / 10.
            i += 1
            
            if len(levels) == 2:
                break
            
            if i >= maxiter:
                break
        
        return levels
    
    def load_output(self, pt='B', prefix='DARE_full_2D'):
        """
        Load results of Geraint's foreground removal runs.
        
        Parameters
        ----------
        pt : str
            Turning point to study (B, C, D)
            
        Returns
        -------
        Stores x, y, z, and 1-2 sigma confidence contours.
        
        """

        i = cols['T_%s' % pt]
        j = cols['nu_%s' % pt]

        x = np.array(readers.readtab('%s/%s_%i_%i_x' % (self.dir, prefix, i, j)))
        y = np.array(readers.readtab('%s/%s_%i_%i_y' % (self.dir, prefix, i, j)))
        
        self.post = np.array(readers.readtab('%s/%s_%i_%i' % (self.dir, prefix, i, j)))
        self.conf = np.array(readers.readtab('%s/%s_%i_%i_cont' % (self.dir, prefix, i, j))).squeeze()
        
        self.z = nu_0_mhz / x - 1.
        self.T = y * 1e3
        
        zz, TT = np.meshgrid(self.z, self.T)
        self.zML = np.ravel(zz)[np.ravel(self.post) == np.max(self.post)]
        self.TML = np.ravel(TT)[np.ravel(self.post) == np.max(self.post)]
        
    def _make_colorbar(self, fig, ax, bounds, ticks=5, label=None, loc='right'):
        """
        Create colorbar flush to right side of axis.
        
        Returns
        -------
        Axis and colorbar objects.
        
        """
        
        left, bottom, right, top = np.ravel(ax.axes.get_position())
        
        if loc == 'right':
            rot = 270
            ori = 'vertical'
            cax = fig.add_axes([right, bottom, 0.025, top-bottom])
        elif loc == 'top':
            rot = 0
            ori = 'horizontal'
            cax = fig.add_axes([left, top, (right-left), 0.025])
            
        cmap = mpl.cm.jet
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        cb = mpl.colorbar.ColorbarBase(cax, norm=norm, orientation=ori)
        if loc == 'right':
            cb.ax.yaxis.set_label_position(loc)
            cb.ax.yaxis.set_ticks_position(loc)
        else:
            
            cb.ax.xaxis.set_label_position(loc)
            cb.ax.xaxis.set_ticks_position(loc)
            cb.ax.xaxis.set_label_coords(0.5, 4.2)
            
        cb.set_ticks(np.linspace(bounds[0], bounds[-1], ticks))
            
        if label is not None:
            cb.set_label(label, rotation=rot)
        
        return ax, cb
        
    def _parse_contours(self, cs):
        """
        Given QuadContourSet, extract redshift, temperature, so that we may
        map it to another space.
        """
        
        pass
                        
    def constrain_B(self, zpts=100, Tpts=100, zlim=(10, 40), Tlim=(-50, 10), 
        plot=False, bins=250, err2d=False, maskDR=False, save_grid=False,
        ne=0.0):
        """
        Constrain background intensity at the Lyman-alpha resonance.
        
        Parameters
        ----------
        bins : int
            Number of bins to use in histogram of 1D Jalpha PDF.
        err2d : bool
            Compute likelihood contours enclosing 68% and 95% of the total
            likelihood? Saves as attribute self.levels.
            
        maskDR : bool
            Use prior knowledge for setting disallowed region? (i.e.,
            assume no exotic heat sources prior to first star formation)
        
        Returns
        -------
        Nothing. Creates attributes likelihood, Jgrid, and Jpdf for further
        analysis. If plot == True, saves axes as attribute too (self.ax).
        """
        
        # Compute constraints on background LyA intensity
        if not hasattr(self, 'Jgrid'):
            Jgrid = self.Jgrid = self.simpl.Jalpha_grid(zpts=zpts, Tpts=Tpts,
                zlim=zlim, Tlim=Tlim, ne=ne)
        else:
            print "Attribute Jgrid exists. Using it."
            
        # Error ellipses
        zB, TB, conf = self.confidence_grid(zpts=zpts, Tpts=Tpts, zlim=zlim,
            Tlim=Tlim)
        
        norm = conf.max()
        conf /= norm
        self.conf = conf
        self.likelihood = conf.copy()
        
        self.zBpts, self.TBpts = zB, TB
        
        # Filter out disallowed region
        self.mask = np.ones_like(conf)
        for i, z in enumerate(zB):
            for j, T in enumerate(TB):
                if T > self.simpl.AbsorptionSignal(z, uv=False):
                    self.mask[i,j] = 0
        
        if maskDR:
            self.likelihood *= self.mask
        
        # Compute 1D marginalized Jalpha and its derivative distributions
                
        hist, bin_edges = np.histogram(np.ravel(self.Jgrid['Ja']),
            weights=np.ravel(self.likelihood), 
            bins=np.logspace(-4, 1, bins))

        hist2, bin_edges2 = np.histogram(np.ravel(self.Jgrid['dJa']),
            weights=np.ravel(self.likelihood), 
            bins=np.logspace(0, 2, bins))
            
        #dJdz = np.ravel(self.Jgrid['dJdz'])
        #dJdz = dJdz[np.isfinite(dJdz)]
        #hist3, bin_edges3 = np.histogram(dJdz[np.isfinite(dJdz)],
        #    weights=np.ravel(self.likelihood)[np.isfinite(dJdz)], 
        #    bins=np.linspace(min(dJdz[np.isfinite(dJdz)]), 
        #    max(dJdz[np.isfinite(dJdz)]), bins))
          
        err_Ja = []  
        err_Ja_dot = []      
        for err in [0.68, 0.95, 0.995]:
            try: 
                err_Ja.append(error_1D(rebin(bin_edges), hist, err))
            except IndexError:
                err_Ja.append(None)
                
            try: 
                err_Ja_dot.append(error_1D(rebin(bin_edges2), hist2, err))
            except IndexError:
                err_Ja_dot.append(None)
            
        bins = rebin(bin_edges)
        bins2 = rebin(bin_edges2)
        
        self.Jpdf = {'bins': bins, 'pdf': hist / hist.max(),
            'err68': err_Ja[0], 'err95': err_Ja[1], 'err99': err_Ja[2],
            'JML': bins[np.argmax(hist)]}
            
        self.dlogJpdf = {'bins': bins2, 'pdf': hist2 / hist2.max(),
            'err68': err_Ja_dot[0], 'err95': err_Ja_dot[1], 'err99': err_Ja_dot[2],
            'JML': bins2[np.argmax(hist2)]}    
    
        if err2d:
            self.levels_B = self._find_sigma_contours(pt='B')
        else:
            self.levels_B = [1e-2, 1e-1]
        
        #if save_grid:
            
        
        if not plot:
            return
        
        # Begin figure
        mp = multipanel(dims=(2,1), left=0.15, right=0.8)
        
        # Plot Jalpha (filled) contours
        mp.contourf(Jgrid['z'], Jgrid['dTb'], np.log10(Jgrid['Ja']), 
            levels=np.linspace(-3., -0.5, 26), extend='both')

        # Plot dlogJalpha (filled) contours
        ax.contourf(Jgrid['z'], Jgrid['dTb'], np.log10(Jgrid['dJa']), 
            levels=np.linspace(-2, 2, 41), extend='both')

        # Add shaded disallowed region
        zarr = np.arange(10., 40., 0.1)
        ax.fill_between(zarr, map(lambda z: self.simpl.AbsorptionSignal(z, uv=False), zarr), 
            1e3 * np.ones_like(zarr), color='gray')  
        ax.plot(zarr, map(lambda z: self.simpl.AbsorptionSignal(z, uv=False), zarr), color='k')
        
        # Limits and labels
        ax.set_xlim(15, 40)
        ax.set_ylim(-50, 10)
        ax.set_xlabel(r'$z_{\mathrm{B}}$')    
        ax.set_ylabel(r'$\delta T_b(z_{\mathrm{B}}) \ [\mathrm{mK}]$')
        
        # Colorbar
        if derivative:
            ax, cb = self._make_colorbar(fig, ax, 
                bounds=np.linspace(-2., 2., 26),
                ticks=6, label=r'$\log_{10} \left(\frac{ d \log J_{\alpha}}{d\log t}\right)$',
                loc='right')
        else:
            ax, cb = self._make_colorbar(fig, ax,
                bounds=np.linspace(-3., -0.5, 26),
                ticks=6, label=r'$\log_{10} \left(J_{\alpha}/J_{21}\right)$',
                loc='right')
        
        # 68% and 95% contours
        cs1 = ax.contour(zB, TB, conf, levels=self.levels_B, 
            linestyles=['--', '-'], linewidths=4, colors='w')
        cs2 = ax.contour(zB, TB, conf, levels=self.levels_B, 
            linestyles=['--', '-'], linewidths=2, colors='k')

        ax.scatter(self.real21['zB'], self.real21['TB'], marker='+', s=150, 
            color='w', lw=5)
        ax.scatter(self.real21['zB'], self.real21['TB'], marker='+', s=150, 
            color='k', lw=2)
        
        pl.draw()
        self.ax = ax
        
    def constrain_C(self, zpts=100, Tpts=100, zlim=(10, 40), Tlim=(-300, 0),
        err2d=False, plot=True, bins=200, maskDR=True):
        """
        Constrain heating rate density.
        """
    
        # Compute constraints on background LyA intensity
        Egrid = self.Egrid = self.simpl.Cheat_grid(zpts=zpts, Tpts=Tpts,
            zlim=zlim, Tlim=Tlim)
    
        # Error ellipses
        zC, TC, conf = self.confidence_grid(zpts=zpts, Tpts=Tpts, zlim=zlim,
            Tlim=Tlim, pt='C')
        
        norm = conf.max()
        conf /= norm
        self.likelihood = conf
    
        # Filter out disallowed region
        if maskDR:
            self.mask = np.ones_like(conf)
            for i, z in enumerate(zC):
                for j, T in enumerate(TC):
                    if T > self.simpl.AbsorptionSignal(z, uv=True):
                        self.mask[i,j] = 0
                        
            self.likelihood *= self.mask
        else:
            self.mask = None
    
        if err2d:
            self.levels_C = self._find_sigma_contours(pt='C')
        else:
            self.levels_C = [1e-2, 1e-1]
            
        # Compute 1D marginalized Jalpha distribution
        hist, bin_edges = np.histogram(np.ravel(self.Egrid['Cheat']),
            weights=np.ravel(self.likelihood),
            bins=np.logspace(50, 53, bins))

        err68 = error_1D(rebin(bin_edges), hist, 0.68)
        err95 = error_1D(rebin(bin_edges), hist, 0.95)
        
        bins = rebin(bin_edges)
        self.Epdf = {'bins': bins, 'pdf': hist / hist.max(),
            'err68': err68, 'err95': err95, 'EML': bins[np.argmax(hist)]}                
        
        if not plot:
            return
            
        # Begin figure
        fig = pl.figure(figsize=(8, 8))
        fig.subplots_adjust(left=0.15, top=0.75)
        ax = fig.add_subplot(111)
    
        # Plot Jalpha (filled) contours
        ax.contourf(Egrid['z'], Egrid['dTb'], np.log10(Egrid['Cheat']), 
            levels=np.linspace(50, 53, 31), extend='both')
    
        # Add shaded disallowed region
        zarr = np.arange(10., 40., 0.1)
        ax.fill_between(zarr, -1e3 * np.ones_like(zarr),
            map(lambda z: self.simpl.AbsorptionSignal(z), zarr), color='gray')
        ax.plot(zarr, map(lambda z: self.simpl.AbsorptionSignal(z), zarr), color='k')
    
        # Limits and labels
        ax.set_xlim(10, 40)
        ax.set_ylim(-350, 0)
        ax.set_xlabel(r'$z_{\mathrm{C}}$')    
        ax.set_ylabel(r'$\delta T_b(z_{\mathrm{B}}) \ [\mathrm{mK}]$')
    
        # Colorbar  
        ax, cb = self._make_colorbar(fig, ax, bounds=np.linspace(50., 53., 31),
            ticks=7, loc='top', 
            label=r'$\log_{10} \left\{ \int_{z_{\mathrm{C}}}^{\infty} \epsilon_{\mathrm{heat}}(z^{\prime}) \frac{dt}{dz^{\prime}} dz^{\prime} \right\} \ \left[\mathrm{erg} \ \mathrm{cMpc}^{-3}\right]$')
    
        # Error ellipses
        zC, TC, conf = self.confidence_grid(zpts=zpts, Tpts=Tpts, 
            zlim=(15, 25), Tlim=(-120, -80), pt='C')
    
        norm = conf.max()
        conf /= norm
            
        # Need to do 68% and 95% rather than lines of constant likelihood
        ax.contour(zC, TC, conf, levels=self.levels_C, linestyles=['--', '-'], 
            linewidths=4, colors='w')
        ax.contour(zC, TC, conf, levels=self.levels_C, linestyles=['--', '-'], 
            linewidths=2, colors='k')
    
        ax.scatter(self.real21['zC'], self.real21['TC'], marker='+', s=150, 
            color='w', lw=5)
        ax.scatter(self.real21['zC'], self.real21['TC'], marker='+', s=150, 
            color='k', lw=2)
    
        #self.load_output()
    
        #ax.contour(noion.z, noion.T, noion.post.T, levels=[noion.conf[0]], 
        #    colors='w', linewidths=4)
        #ax.contour(noion.z, noion.T, noion.post.T, levels=[noion.conf[0]], 
        #    colors='k', linewidths=2)
        #
        #ax.contour(ion.z, ion.T, ion.post.T, levels=[ion.conf[0]], 
        #    colors='w', linewidths=4, linestyles='--')
        #ax.contour(ion.z, ion.T, ion.post.T, levels=[ion.conf[0]], 
        #    colors='k', linewidths=2, linestyles='--')    
    
        pl.draw()
        self.ax = ax            
    
    def overplot_Jlw(self, Jlw=[1e-2, 1e-1, 2e-1]):
        """
        Overplot lines of constant Ja / J21 for the given Lyman-Werner fluxes.
        
        Assumes sources have flat spectra between Lyman-alpha and the Lyman 
        edge.
        
        Uses self.ax instance.
        
        Parameters
        ----------
        levels : float, list
            Lyman-Werner fluxes in units of J21.
        
        """
        
        grid = self.Jgrid
        
        if type(levels) is not list:
            levels = [levels]

        levels = np.array(levels)
        levels /= (E_LL - E_LyA) / (E_LL - 11.2)    
                    
        self.ax.contour(grid['z'], grid['dTb'], np.log10(grid['Ja']), 
            levels=np.log10(levels), colors='w', 
            linestyles=['-', '-'], linewidths=5)
        self.ax.contour(grid['z'], grid['dTb'], np.log10(grid['Ja']), 
            levels=np.log10(levels), colors='k', 
            linestyles=['-', '-'], linewidths=2)    
        
        pl.draw()
        
            