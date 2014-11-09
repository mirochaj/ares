"""

ModelGrid.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Dec  5 15:49:16 MST 2013

Description: For analyzing model grids.

"""

import numpy as np
import time, copy, os, pickle
import matplotlib.pyplot as pl
#from ..simple import Interpret21cm
from ..util import labels#, default_errors
from ..physics.Constants import cm_per_mpc
from ..inference.ModelGrid import ModelGrid as iMG
from ..util.Stats import Gauss1D, GaussND, error_1D, rebin

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1

class ModelGrid(object):
    """Create an object for setting up, running, and analyzing model grids."""
    def __init__(self, prefix, **kwargs):
        """
        Parameters
        ----------
        grid : str
            Name of file containing output of ares.inference.ModelGrid.
        """

        # Instance of ares.inference.ModelGrid
        grid = iMG(prefix=prefix, **kwargs)
        
        if grid.is_restart and rank == 0:
            print "WARNING: This model grid is incomplete! ",
            print "This will likely cause problems in the analysis."
        
        # Instance of ndspace.ModelGrid
        self.grid = grid.grid
        
        if hasattr(grid, 'extras'):
            self.extras = grid.extras
        
        try:
            self.bgrid = iMG(grid='%s.blobs.hdf5' % prefix, **kwargs)
        except IOError:
            print "%s.blobs.hdf5 not found" % prefix
        
    def __getitem__(self, name):
        return self.grid[name]

    @property
    def reference_pars(self):
        """
        Parameter file for the reference model, which will use all defaults
        if self.reference_pars is not set by hand.
        """
        if not hasattr(self, '_reference_pars'):
            self._reference_pars = {}
    
        return self._reference_pars
    
    @reference_pars.setter
    def reference_pars(self, value):
        self._reference_pars = value

    @property
    def reference_model(self):
        if not hasattr(self, '_reference_model'):
            self.reference_pars['track_extrema'] = True
                
            # Run a simulation
            sim = glorb.run.Simulation(load_sim=True, 
                **self.reference_pars)
            if not sim.found_sim:    
                print "Generating reference model (only need to do this once)..."
                sim.run()
                
            # Save reference model as analysis.Synthetic21cm class instance
            self._reference_model = glorb.analysis.Synthetic21cm(sim=sim)            
            
            # Pull out "true" turning point positions
            zB_act, TB_act = self._reference_model.turning_points['B']
            zC_act, TC_act = self._reference_model.turning_points['C']
            zD_act, TD_act = self._reference_model.turning_points['D']
            
            # Initialize Extracted21cm instance
            #Extracted21cm.__init__(self, signal=anl)
                        
        return self._reference_model
        
    @property
    def refB(self):
        return self.reference_model.turning_points['B']
        
    @property
    def refC(self):
        return self.reference_model.turning_points['C']
            
    @property
    def refD(self):
        return self.reference_model.turning_points['D']
    
    @property
    def zB(self):
        if hasattr(self, 'grid_B'):
            return self.grid_B['B'][...,0]
        else:
            return self['B'][...,0]
    
    @property
    def TB(self):
        if hasattr(self, 'grid_B'):
            return self.grid_B['B'][...,1]
        else:
            return self['B'][...,1]
            
    @property
    def zC(self):
        return self['C'][...,0]
    
    @property
    def TC(self):
        return self['C'][...,1]
    
    @property
    def zD(self):
        return self['D'][...,0]
    
    @property
    def TD(self):
        return self['D'][...,1]
    
    @property
    def errors(self):
        if not hasattr(self, '_errors'):
            self._errors = default_errors.copy()
        
        return self._errors
    
    @errors.setter
    def errors(self, value):
        self._errors = default_errors.copy()
        
        for key in value:
            data = np.diag(value[key])
            
            if len(data.shape) == 2:
                self._errors.update({key: data})
            else:
                self._errors.update({key: value[key]})

    @property
    def interpret(self):
        if not hasattr(self, '_interpret'):
            self._interpret = Interpret21cm()
                        
        return self._interpret
                        
    def extract_models(self, zerr_B=0.1, nu=0.68):
        """
        Determine which models which satisfy a given turning point criterion.
        
        So far, just turning point B.
        
        Parameters
        ----------
        zerr_B : float
            Percent error (relative to reference model) in turning point B
            position.
            
        Returns
        -------
        A dictionary containing all models satisfying given criterion.
        
        """
        
        zB, TB = self.reference_model.turning_points['B']
        ztmp = np.linspace(zB-10, zB+10, 10000)
        zpdf = Gauss1D(ztmp, [0, 1, zB, zerr_B*zB])
        err = np.mean(error_1D(ztmp, zpdf, nu=nu))
        
        acceptable_models = {}
        for loc, value in np.ndenumerate(self.grid.grid['B']):

            # Ignore brightness temperatures
            if loc[-1] == 1:
                continue

            if not ((value >= zB - err) and (value <= zB + err)):
                continue

            acceptable_models[loc[0:-1]] = {'zB': value}
            for key in self.extras[loc[0:-1]].keys():
                acceptable_models[loc[0:-1]][key] = self.extras[loc[0:-1]][key]

        return acceptable_models, err

    def set_limits(self, models, fields=['Ja']):
        """
        Given set of models, set min/max limits on evolution in various things.
        """
        
        zB, TB = self.reference_model.turning_points['B']
        #zC, TC = self.reference_model.turning_points['C']

        results = {}
        for field in fields:
            results[field] = {}

        for field in fields:

            # Compute Lyman-alpha evolution for acceptable models
            JB = []
            zcommon = np.linspace(20, 50, 200)
            Jmin, Jmax = np.inf * np.ones_like(zcommon), np.zeros_like(zcommon)
            for i, model in enumerate(models.keys()):

                J = np.zeros_like(zcommon)
                this_model = models[model]
                for j, redshift in enumerate(zcommon):
                    J[j] = np.interp(redshift, this_model['z'][-1::-1],
                        this_model['Ja'][-1::-1])
            
                Jmin = np.minimum(Jmin, J)
                Jmax = np.maximum(Jmax, J)
            
                JB.append(np.interp(zB, this_model['z'][-1::-1],
                    this_model['Ja'][-1::-1]))
                
        return zcommon, np.array(JB), Jmin, Jmax
        
    def likelihood(self, z, T, pt='B'):
        """
        Compute likelihood for point (z, T).
        
        Currently assumes simplest 1-D Gaussian (independent) errors.
        
        Parameters
        ----------
        z : float
            Redshift of point of interest
        T : float
            Corresponding brightness temperature
        zerr : float
            1-sigma error on the redshift measurement.
        Terr : float
            1-sigma error on the brightness temperature (mK). 
 
        Returns
        -------
        Likelihood!

        """

        cov = self.errors[pt]

        z_in, T_in = self.reference_model.turning_points[pt]
        
        mu = np.array([z_in, T_in])

        return GaussND(np.array([z, T]), mu, cov)
        
    def likelihood_grid(self, tp):
        
        L = np.zeros(self.grid.shape)
        z, dTb = self.grid[tp][...,0], self.grid[tp][...,1]

        for index, red in np.ndenumerate(self.grid[tp][:,:,0]):
            L[index] = self.likelihood(red, dTb[index], pt=tp)
                
        return L        

    @property
    def to_solve(self):
        if not hasattr(self, '_to_solve'):
            self._to_solve = self.load_balance()
        
        return self._to_solve
        
    def read_extras(self, fn):
        
        output = {}
        
        f = open('%s' % fn, 'rb')     
        
        while True:
            try:
                output.update(pickle.load(f))
            except EOFError:
                break
        f.close()
                
        self.extras = output
        
        if rank == 0:
            print 'Read %s' % fn
            
    def scatter_turning_points(self, Lscale=False, ax=None, fig=1, 
        slc=None, box=False, track=False, include='BCD', **kwargs):
        """
        Show occurrences of turning points B, C, and D for all models in
        (z, dTb) space, with points color-coded to likelihood.
        
        Parameters
        ----------
        err : dict
            Errors on each turning point, sorted in (z, dTb) pairs.
            Example: {'B': [0.3, 10], 'C': [0.2, 10], 'D': [0.2, 10]}
        Lscale : bool
            Color-code points by likelihood?
            
        """
    
        if ax is None:
            gotax = False
            fig = pl.figure(fig)
            ax = fig.add_subplot(111)
        else:
            gotax = True
        
        self.L = {}
        for tp in list(include):
            
            if slc is not None:
                axis, z = self.grid.slice(self.grid[tp][...,0], slc)
                axis, dTb = self.grid.slice(self.grid[tp][...,1], slc)
            else:
                z, dTb = self.grid[tp][...,0], self.grid[tp][...,1]
            
            if (slc is None) and Lscale:
                L = self.likelihood_grid(tp)
                self.L[tp] = L.copy()
                
                L_to_plot = np.array(L_to_plot)
                L_to_plot /= L_to_plot.max()

                ax.scatter(z_to_plot, T_to_plot,
                    c=L_to_plot, edgecolor='none', **kwargs)

            elif (slc is not None) and Lscale:
                
                L = np.zeros_like(z)
                for i, red in enumerate(z):
                    L[i] = self.likelihood(red, dTb[i], pt=tp)
                        
                self.L[tp] = L.copy()  
                
                ax.scatter(z, dTb, c=self.L[tp], edgecolor='none', **kwargs)

            else:
                
                if box:
                    
                    # Bin centers
                    zz = np.linspace(z.min(), z.max(), 100)
                    dz = np.diff(zz)[0]
                    
                    y1 = np.zeros_like(zz)
                    y2 = np.zeros_like(zz)
                    for i, red in enumerate(zz):
                        condition = np.logical_and(z > (red - 0.5 * dz),
                                                   z <= (red + 0.5 * dz))
                        hist, edges = np.histogram(dTb[condition].ravel(), 
                            bins=2000)
                        
                        y1[i] = edges.min()
                        y2[i] = edges.max()

                    ax.plot(zz, y1, **kwargs)
                    if not track:
                        ax.plot(zz, y2, **kwargs)
                else:
                    ax.scatter(z, dTb, **kwargs)
                    
        ax.set_xlim(5, 45)
        ax.set_ylim(-300, 50)            
    
        pl.draw()
    
        return ax
        
    def analytic_C(self, ax=None, err=None, L=None, nu=[0.68, 0.95, 0.99],
        bins=20, density=True, **kwargs):
        """
        Constrain analytic limits for heating rate and temperature at C.
        
        nu : float
            Confidence interval onto which we'll project analytic results.
        """
        
        if L is None:
            L = self.likelihood_grid('C', err=err)

        z = self.grid['C'][...,0]
        dTb = self.grid['C'][...,1]
        h_1 = self.bgrid['C'][...,list(self.bgrid.grid.higher_dimensions).index('igm_h_1')]

        if ax is None:
            
            # Must convert rate coefficients to proper rates
            mult = np.array([self.interpret.cosm.nH(z) * h_1 * cm_per_mpc**3, 
                             np.ones_like(z)])
            
            ax = self.posterior_pdf(['igm_heat', 'Tk'], L, multiplier=mult,
                contour=True)

        nu, levels = self.confidence_regions(L, nu=nu)

        heat_lo = self.heat_lo = np.zeros(self.grid.shape)
        heat_hi = self.heat_hi = np.zeros(self.grid.shape)
        Tk_lo = self.Tk_lo = np.zeros(self.grid.shape)
        Tk_hi = self.Tk_hi = np.zeros(self.grid.shape)
        for i, coords in enumerate(self.grid.coords):
            
            heat_lo[coords], heat_hi[coords] = \
                self.interpret.heating_rate_C(z[coords], dTb[coords])

            Tk_lo[coords] = self.interpret.cosm.Tgas(z[coords])
            Tk_hi[coords] = \
                self.interpret.Ts(z[coords], dTb[coords])

        #ax.contour(heat_lo, Tk_lo,
        #    L.T / L.max(), levels, **kwargs)
        ax.contour(heat_hi, Tk_hi, 
            L.T / L.max(), levels, **kwargs)
        
        pl.draw()
        
        return ax

    def posterior_pdf(self, pars, likelihood=None, turning_point='C', ax=None, 
        fig=1, plot=True, multiplier=None,
        nu=[0.68, 0.95, 0.99], slc=None, overplot_nu=False, density=True, 
        color_by_nu=False, contour=True, filled=True, 
        bins=20, analytic=False, xscale='log', yscale='log', **kwargs):
        """
        Compute posterior PDF for supplied parameters. 
        
        If len(pars) == 2, plot 2-D posterior PDFs. If len(pars) == 1, plot
        1-D marginalized PDF.
        
        Parameters
        ----------
        pars : str, list
            Name of parameter or list of parameters to analyze.
        likelihood : np.ndarray
            Likelihood (2D) corresponding to supplied errors etc.
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
        analytic : bool
            Overplot analytic limits?
            
        Returns
        -------
        Either a matplotlib.Axes.axis object or a nu-sigma error-bar, 
        depending on whether we're doing a 2-D posterior PDF (former) or
        1-D marginalized posterior PDF (latter).
            
        """

        if ax is None and plot:
            gotax = False
            fig = pl.figure(fig)
            ax = fig.add_subplot(111)
        else:
            gotax = True
            
        if type(pars) not in [list, tuple]:
            pars = [pars]
            
        p = []
        b = []
        for par in pars:
            if par in self.grid.axes_names:
                p.append(self.grid.meshgrid(par))
                b.append(self.grid.axes[self.grid.axes_names.index(par)].values)
            elif par in self.bgrid.grid.higher_dimensions:
                j = list(self.bgrid.grid.higher_dimensions).index(par)
                axmesh = np.zeros(self.bgrid.grid.shape)
                for i, coords in enumerate(self.bgrid.grid.coords):
                    axmesh[coords] = self.bgrid.grid[turning_point][coords][j]
                
                p.append(axmesh)
                b.append(bins)
                
        L = likelihood = self.likelihood_grid(turning_point)        
                                                          
        if len(pars) == 1:

            hist, bin_edges = \
                np.histogram(p[0].ravel(),
                weights=likelihood.ravel(), density=density, bins=b[0])
            
            bc = rebin(bin_edges)
            
            try:
                mu, sigma = bc[hist == hist.max()], error_1D(bc, hist, nu=nu)
            except ValueError:
                mu, sigma = bc[hist == hist.max()], error_1D(bc, hist, nu=nu[0])
            
            if plot:

                ax.plot(bc, hist / hist.max(), 
                    drawstyle='steps-mid', **kwargs)
                ax.set_xscale('log')
                ax.set_xlabel(labels[pars[0]])
                ax.set_ylabel('PDF')

                if overplot_nu:
                    mi, ma = ax.get_ylim()

                    ax.plot([mu - sigma[0]]*2, [mi, ma], color='k', ls=':')
                    ax.plot([mu + sigma[1]]*2, [mi, ma], color='k', ls=':')

                return ax

            else:
                return mu, sigma

        elif type(pars) is list:

            #x = np.zeros_like(p)
            #for i, coords in enumerate(self.grid.coords):
            #    for j, par in enumerate(pars):
            #        x[j][coords] = p[j][coords] * multiplier[j][coords]
                
            #p = x#np.array(p) #* np.array(multiplier)
             
            if multiplier is not None:
                x = np.array(p) * multiplier 
                p = x        
            else:
                x = np.array(p)
                p = x    
                        
            if color_by_nu:
                
                # Get likelihood contours (relative to peak) that enclose
                # nu-% of the area
                nu, levels = self.get_levels(likelihood, nu=nu)
                
                if contour:   
                    if filled:
                        ax.contourf(x[0], x[1], likelihood / likelihood.max(), 
                            levels, **kwargs)
                    else:
                        ax.contour(x[0], x[1], likelihood / likelihood.max(),
                            levels, **kwargs)
                else:
                    ax.scatter(p[0].ravel(), p[1].ravel(),
                        c=c, edgecolor='none', **kwargs)

            else:
                if contour:
                    if filled:
                        ax.contourf(x[0], x[1], likelihood, **kwargs)
                    else:
                        ax.contour(x[0], x[1], likelihood, **kwargs)
                else:
                    levels = likelihood.ravel()
                    ax.scatter(p[0].ravel(), p[1].ravel(),
                        c=levels, edgecolor='none', **kwargs)

            ax.set_xscale(xscale)
            ax.set_yscale(yscale)
            
            if pars[0] is not 'dTb':
                ax.set_xlim(p[0][p[0] >= 0].min(), p[0].max())
            else:
                ax.set_xlim(p[0].min(), p[0].max())
            
            if pars[1] is not 'dTb':
                ax.set_ylim(p[1][p[1] > 0].min(), p[1].max())
            else:    
                ax.set_ylim(p[1].min(), p[1].max())
            
            if pars[0] in labels:
                ax.set_xlabel(labels[pars[0]])
            else:
                ax.set_xlabel(pars[0])
                
            if len(pars) == 2:
                if pars[1] in labels:
                    ax.set_ylabel(labels[pars[1]])
                else:
                    ax.set_ylabel(pars[1])
            
            pl.draw()
            
            return ax
            
        else:
            raise ValueError('dont know how to handle > 2 dims yet!')
        
    def get_levels(self, L, nu=[0.68, 0.95, 0.99]):
        """
        Return levels corresponding to input nu-values, and assign
        colors to each element of the likelihood.
        """

        nu, levels = self.confidence_regions(L, nu=nu)

        tmp = L.ravel() / L.max()

        return nu, levels

    def confidence_regions(self, L, nu=[0.68, 0.95, 0.99]):
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
    
    
    
                
    