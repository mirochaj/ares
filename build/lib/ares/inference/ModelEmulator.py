"""

ModelEmulator.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Thu Oct 12 17:10:54 PDT 2017

Description: 

"""

import time
import copy
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as pl
import matplotlib.pyplot as plt
from ..physics.Constants import nu_0_mhz
from ..analysis import Global21cm as aGS
from ..util.Aesthetics import labels
from ..analysis import ModelSet, MultiPanel
from ..analysis.MultiPlot import add_master_legend

try:
    import emupy
except ImportError:
    pass

#try:
#    import astropy.stats as astats
#except ImportError:
#    pass    

try:
    import sklearn.gaussian_process as gp
except ImportError:
    pass    

class ModelEmulator(object): # pragma: no cover
    def __init__(self, tset):
        if type(tset) == str:
            self.tset = ModelSet(tset)
        else:
            assert isinstance(tset, ModelSet)
            self.tset = tset

    @property
    def vset(self):
        if not hasattr(self, '_vset'):
            raise AttributeError('Must set validation set `vset` by hand!')
        return self._vset
        
    @vset.setter
    def vset(self, value):
        if type(value) == str:
            self._vset = ModelSet(value)
        else:
            assert isinstance(value, ModelSet)
            self._vset = value
            
    def show_samples(self, pars=None, downsample=None, **kwargs):
        if pars is None:
            pars = self.tset.parameters
            
        if downsample is not None:
            skip = 0
            stop = downsample
            skim = None    
        else:
            skip = stop = skim = None    
            
        mp = self.tset.TrianglePlot(self.tset.parameters, scatter=True, 
            oned=False, skip=skip, stop=stop, skim=skim, **kwargs)
            
        return mp        
    
    def train(self, ivars=None, field='dTb', ivar_tol=1e-2, use_pca=False, 
        nmodes=10, method='poly', lognorm_data=False, lognorm_pars=True,
        downsample=None, verbose=False):
        """
        Run the emulator, obtain an object capable of new predictions.
        """
        
        Ns = self.tset.chain.shape[0]
        if downsample is not None:
            Ns = min(Ns, int(downsample))
            
        sample_ids = np.arange(0, Ns)
        
        # Set independent variables for training
        ivar_raw = np.array(self.tset.get_ivars(field)).squeeze()
        if ivars is None:
            ivar_slc = np.ones_like(ivar_raw)
        else:
            ivar_slc = np.zeros_like(ivar_raw)
            
            for ivar in ivars:
                i = np.argmin(np.abs(ivar - ivar_raw))
                if abs(ivar_raw[i] - ivar) > ivar_tol:
                    continue
                    
                ivar_slc[i] = 1
                
        # Should be equal to ivars if it was supplied        
        self.ivar_slc = ivar_slc
        ivar_arr = self.ivar_arr = ivar_raw[ivar_slc == 1]
                
        # Read-in the training data and slice it up
        all_data_raw = self.tset.ExtractData(field)[field]
        all_data = all_data_raw[:,ivar_slc == 1]
        dat_grid = np.array([all_data[i] for i in sample_ids])
        par_grid = np.array([self.tset.chain[i,:] for i in sample_ids])
        
        if lognorm_pars:
            self.lognorm_pars = True
            par_grid = np.log(par_grid)
        else:
            self.lognorm_pars = False
    
        # Initialize Emulator and its variables
        # see help(E) for details on these parameters and their default values
        E = self.E = emupy.Emu()

        # Specify number of samples in training data
        E.N_samples = Ns

        # Specify number of model parameters
        E.N_params = len(self.tset.parameters)

        # Specify number of data elements
        # For PS this is 2-D, GS just 1-D, have to flatten in any case
        E.N_data = len(ivar_arr)

        # Specify usage of pca or not
        E.use_pca = use_pca

        # If using pca, specify number of eigenmodes we want to keep during truncation
        E.N_modes = nmodes

        # Choose regression model for interpolation functions
        E.reg_meth = method

        # Choose to log normalize the data before taking covariance
        E.lognorm = lognorm_data # have negative values!

        # Whiten the training data y-values
        E.data_whiten = False

        # Calculate fiducial (or average) data properties
        E.fid_data = np.median(dat_grid, axis=0)
        E.fid_grid = np.median(par_grid, axis=0)

        # Sphere training data x values
        E.sphere(par_grid, fid_grid=E.fid_grid, save_chol=True, norotate=True)

        # Perform KLT to calculate eigenmodes
        E.klt(dat_grid, normalize=True)

        ## Setup Gaussian Process kwargs
        # Setup a squared exponential kernel with diagonal noise hyperparameter
        kernel = gp.kernels.RBF(length_scale=np.ones(E.N_params)) \
               + gp.kernels.WhiteKernel(noise_level=1e-6)

        # How many times to restart MLE search of hyperparameters
        n_restarts_optimizer = 10

        # Which optimizer to use
        optimizer='fmin_l_bfgs_b'

        # pack into a dictionary
        gp_kwargs = {'kernel':kernel, 'n_restarts_optimizer':n_restarts_optimizer, 
            'optimizer':optimizer}
        E.gp_kwargs = gp_kwargs

        E.train(dat_grid, par_grid, verbose=verbose)
                        
    def predict(self, vals=None, **kwargs):
        
        if vals is None:
            assert kwargs != {}
            
            vals = []
            for i, par in enumerate(self.tset.parameters):
                if par not in kwargs:
                    # Will already have been log-ified if lognorm_pars==True
                    vals.append(self.E.fid_grid[i])
                else:
                    if self.lognorm_pars:
                        vals.append(np.log(kwargs[par]))
                    else:
                        vals.append(kwargs[par])
                    
            vals = np.array(vals)
        else:
            if self.lognorm_pars:
                vals = np.log(vals)
        
        recon, recon_err, recon_cov, recon_w, recon_werr = \
            self.E.predict(vals, output=True)
        
        return recon
        
    def validate(self, pars=None, ivars=None, field='dTb', mp_kwargs={}, fig=1, 
        mp=None, Nvals=10, scatter=True, **kwargs):
        """
        Compare the emulator predictions with the results from a high 
        resolution grid.
    
        """
    
        if pars is None:
            defaults = self.E.fid_grid
            if self.lognorm_pars:
                defaults = np.exp(defaults)
        else:
            raise NotImplemented('')
    
        if ivars is None:
            ivars = self.ivar_arr
        else:
            ivars = np.array(ivars)
        
        dims = len(ivars), len(self.tset.parameters)
    
        if mp is None:
            got_mp = False
            mp = MultiPanel(dims=dims, fig=fig, **mp_kwargs)
        else:
            got_mp = True
        
        # Loop over parameters, plot 'true data' and prediction from training
        # set for input `ivars`. Basically, series of 1-D parameter studies.
        # Key is making sure the validation set holds all other dimensions
        # at the right level (i.e., that used in the training set).
        for i, par in enumerate(self.tset.parameters):
    
            # Find where this parameter lives in the validation grid.
            ll = list(self.vset.parameters).index(par)
            
            # Set up range of x-values that all lie within training set
            mi = np.min(self.tset.chain[:,i])
            ma = np.max(self.tset.chain[:,i])
            par_vals = np.logspace(np.log10(mi), np.log10(ma), Nvals)
    
            # Determine which elements from validation grid to keep.
            # Just need to worry about higher dimensions, i.e., not `par`
            keep = np.ones(self.vset.chain.shape[0])
            for j, par2 in enumerate(self.vset.parameters):
                
                if par2 == par:
                    continue
                
                # Only select elements from validation grid that
                # used the same input value as training set
                if par2 not in self.tset.parameters:
                    if par2 in self.tset.base_kwargs:
                        tval = self.tset.base_kwargs[par2]
                    else:
                        tval = np.median(self.vset.unique_samples[j])
                        
                    kk = np.argmin(np.abs(self.vset.unique_samples[j] - tval))                    
                else:
                    ii = list(self.tset.parameters).index(par2)
                    tval = np.median(self.tset.unique_samples[ii])
                    kk = np.argmin(np.abs(self.vset.unique_samples[j] - tval))                    
                
                keep *= self.vset.chain[:,j] == self.vset.unique_samples[j][kk]
    
            # Indices of elements to keep
            ikeep = np.argwhere(keep == 1)
        
            # Use indices to create a new object with only the samples we want
            slc = self.vset.SliceByElement(ikeep)
            
            # Further refine based on the indep. variable values we want
            dat = slc.ExtractData(field)[field][:,self.ivar_slc == 1]
            raw_ivars = slc.get_ivars(field).squeeze()

            ###
            ## PREDICTIONS FROM TRAINING SET.
            ## Part I. Setup array of (samples, parameters) to input
            ##         to E.predict
            ###
            
            # Convert to an array with the shape of a chain, i.e.,
            # (number of models, number of parameters)
            cross_val = []
            for j, par2 in enumerate(self.tset.parameters):
                if par2 == par:
                    cross_val.append(par_vals)
                    continue
                       
                cross_val.append(np.ones(par_vals.size) * defaults[j])
            
            cross_val = np.array(cross_val).T
                        
            ##
            # USE TRAINING SET TO MAKE PREDICTION
            ##
            # The log-ness of these values will be handled within `predict`
            recon = self.predict(cross_val)
            
            # Plot as function of `ivar`
            for k, ivar in enumerate(ivars):
                 m = mp.axis_number(k, i)
                 
                 # Make sure the ivar we want is actually available.
                 l = np.argmin(np.abs(ivar - self.ivar_arr))
                 this_v = self.ivar_arr[l]

                 if abs(this_v - ivar) > 1e-3:
                     continue

                 # Plot right answers from validation set
                 ii = list(self.vset.parameters).index(par)
                 x = self.vset.chain[keep == 1,ii]
                 
                 # Must find index 'll' that maps to reduced ivar array
                 ll = np.argmin(np.abs(raw_ivars[self.ivar_slc == 1] - ivar))
                                  
                 y = dat[keep == 1,ll]
                 s = np.argsort(x)
                 mp.grid[m].plot(x[s], y[s], color='k', ls='-')
            
                 # Plot predictions
                 x2, y2 = cross_val[:,i], recon[:,l]
                 s2 = np.argsort(x2)
                 
                 # Clean up
                 if scatter:
                     mp.grid[m].scatter(x2[s2], y2[s2], **kwargs)    
                 else:
                     mp.grid[m].plot(x2[s2], y2[s2], **kwargs)    
                     
                 mp.grid[m].set_xscale('log')
                 mp.grid[m].set_yscale('linear')
            
                 if m in mp.left and (i == 0) and (not got_mp):
                     # Should get name automatically too
                     mp.grid[m].annotate(r'$z=%i$' % round(ivar), (0.05, 0.95),
                         ha='left', va='top', xycoords='axes fraction')
    
        if not got_mp:
            for i in range(len(self.tset.parameters)):
                mp.grid[i].set_xlabel(self.tset.parameters[i])
            
            for i in mp.left:
                mp.grid[i].set_ylabel(r'$\delta T_b$')
            
        for i in mp.elements.ravel():
            if i not in mp.bottom:    
                mp.grid[i].set_xticklabels([])
            if i not in mp.left:
                mp.grid[i].set_yticklabels([])
            
        mp.align_labels(0.3, 0.3)
        
        return mp
        
    def validate_recon(self, N=10, field='dTb', ax=None, fig=1, 
        z_to_freq=True, **kwargs):
            
        if ax is None:
            gotax = False
            fig = pl.figure(fig)
            ax = fig.add_subplot(111)
        else:
            gotax = True    
            
        zarr = self.vset.get_ivars('dTb')
        dTb =  self.vset.ExtractData('dTb')['dTb']

        samples = np.random.randint(0, self.vset.chain.shape[0], size=N)

        colors = ['b', 'r', 'g', 'c', 'k', 'm', 'y', 'c']
        lc = len(colors)
        
        for k, i in enumerate(samples):
            if z_to_freq:
                ax.plot(nu_0_mhz / (1. + zarr), dTb[i], 
                    color=colors[k], ls='-', alpha=0.5)
            else:    
                ax.plot(zarr, dTb[i], color=colors[k%lc], ls='-', alpha=0.5)

            # Training set and validation set parameters in different order!
            chain = []
            for j, par in enumerate(self.tset.parameters):
                l = list(self.vset.parameters).index(par)
                chain.append(self.vset.chain[i,l])

            chain = np.array(chain)
            
            if z_to_freq:
                ax.plot(nu_0_mhz / (1. + self.ivar_arr), self.predict(chain)[0], 
                    color=colors[k%lc], ls='--')
            else:    
                ax.plot(self.ivar_arr, self.predict(chain)[0],
                    color=colors[k%lc], ls='--')
            
        ax.set_ylabel(labels[field])    

        if z_to_freq:
            ax.set_xlabel(labels['nu'])
        else:    
            ax.set_xlabel(r'$z$')
            
        return ax
    
    def create_simulator(self, defaults):
        """
        Make an object that acts like an instance of some ares.simulations
        class.
        """
        
        # Needs to know parameter names and indices
        sim = SimulationEmulator(self.E, self.tset.parameters, 
            self.ivar_arr, defaults)
    
        return sim
    
  
class SimulationEmulator(object):
    def __init__(self, emulator, pars, ivars, defaults):
        self.E = emulator
        self.parameters = pars
        self.ivars = ivars
        self.defaults = np.array([defaults[par] for par in pars])

    def __call__(self, **kwargs):
        # This actually mimics the __init__ method of simulation objects

        arr = self.defaults.copy()
        for i, par in enumerate(self.parameters):
            if par not in kwargs:
                continue
            arr[i] = kwargs[par]

        self.arr = arr

    def run(self):
        recon = self.E.predict(self.arr)

        # Create 'history' attribute.
        hist = {'z': self.ivar_arr, 'dTb': recon.squeeze()}
        
        # How to make the routines accessible? Will inheriting aGS work?
        self.history = aGS(history=hist)



