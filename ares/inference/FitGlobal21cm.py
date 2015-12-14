"""

ModelFit.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon May 12 14:01:29 MDT 2014

Description: 

"""

import numpy as np
from ..util.PrintInfo import print_fit
from .ModelFit import ModelFit, LogPrior
from ..physics.Constants import nu_0_mhz
import gc, os, sys, copy, types, time, re
from ..simulations import Global21cm as simG21
from ..analysis import Global21cm as anlGlobal21cm
from ..analysis.InlineAnalysis import InlineAnalysis
from ..simulations import Global21cm as simGlobal21cm
from ..util.SetDefaultParameterValues import _blob_names, _blob_redshifts
 
try:
    import cPickle as pickle
except:
    import pickle

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1

extrema_warning = "WARNING: Did you forget to set track_extrema=True?"
extrema_warning += " It's not too late to modify base_kwargs!"

def_kwargs = {'verbose': False, 'progress_bar': False,
    'one_file_per_blob': True}

class loglikelihood:
    def __init__(self, xdata, ydata, error, parameters, is_log,
        base_kwargs, priors={}, prefix=None, blob_names=None, 
        blob_ivars=None, blob_funcs=None, turning_points=None):
        """
        Computes log-likelihood at given step in MCMC chain.

        Parameters
        ----------

        """

        self.parameters = parameters
        self.is_log = is_log

        self.base_kwargs = base_kwargs

        self.prefix = prefix 

        self.blob_names = blob_names
        self.blob_ivars = blob_ivars
        self.blob_funcs = blob_funcs
        self.turning_points = turning_points
        
        # Sort through priors        
        priors_P = {}   # parameters
        priors_B = {}   # blobs

        p_pars = []
        b_pars = []
        for key in priors:
            # Priors on model parameters
            if len(priors[key]) == 3:
                p_pars.append(key)
                priors_P[key] = priors[key]

            elif len(priors[key]) == 4:
                b_pars.append(key)
                priors_B[key] = priors[key]
            
            # Should set up a proper Warnings module for this sort of thing
            if key == 'tau_e' and len(priors[key]) != 4:
                if rank == 0:
                    print 'Must supply redshift for prior on %s!' % key
                MPI.COMM_WORLD.Abort()

        self.logprior_P = LogPrior(priors_P, self.parameters, self.is_log)
        self.logprior_B = LogPrior(priors_B, b_pars)

        if self.turning_points:
            
            nu = [xdata[i] for i, tp in enumerate(self.turning_points)]
            T = [ydata[i] for i, tp in enumerate(self.turning_points)]
            
            self.xdata = None
            self.ydata = np.array(nu + T)
        else:
            self.xdata = xdata
            self.ydata = ydata
            
        self.error = error

        #self.is_cov = False        
        #if len(self.errors.shape) > 1:
        #    self.is_cov = True
        #    self.Dcov = np.linalg.det(self.errors)
        #    self.icov = np.linalg.inv(self.errors)
        #

    @property
    def blank_blob(self):
        if not hasattr(self, '_blank_blob'):
    
            self._blank_blob = []
            for i, group in enumerate(self.blob_names):
                if self.blob_ivars[i] is None:
                    self._blank_blob.append([np.inf] * len(group))
                else:
                    arr = np.ones([len(group), self.blob_ivars[i].size])
                    self._blank_blob.append(arr * np.inf)
    
        return self._blank_blob
    
    def __call__(self, pars, blobs=None):
        """
        Compute log-likelihood for model generated via input parameters.

        Returns
        -------
        Tuple: (log likelihood, blobs)

        """

        kwargs = {}
        for i, par in enumerate(self.parameters):
            if self.is_log[i]:
                kwargs[par] = 10**pars[i]
            else:
                kwargs[par] = pars[i]

        # Apply prior on model parameters first (dont need to generate signal)
        lp = self.logprior_P(pars)
        if not np.isfinite(lp):
            return -np.inf, self.blank_blob

        # Run a model and retrieve turning points
        kw = self.base_kwargs.copy()
        kw.update(kwargs)

        try:
            sim = self.sim = simG21(**kw)
            sim.run()
            
            #sim.run_inline_analysis()
            
            tps = sim.turning_points
                                        
        # Timestep weird (happens when xi ~ 1)
        except SystemExit:
            
            #sim.run_inline_analysis()
            tps = sim.turning_points
                 
        # most likely: no (or too few) turning pts
        #except ValueError:                     
        #    # Write to "fail" file
        #    if not self.burn:
        #        f = open('%s.fail.%s.pkl' % (self.prefix, str(rank).zfill(3)), 'ab')
        #        pickle.dump(kwargs, f)
        #        f.close()
        #
        #    del sim, kw, f
        #    gc.collect()
        #
        #    return -np.inf, self.blank_blob

        # Apply priors to blobs
        blob_vals = []
        for key in self.logprior_B.priors:

            if not hasattr(sim, 'blobs'):
                break
            
            z = self.logprior_B.priors[key][3]

            i = self.blob_names.index(key) 
            j = self.blob_redshifts.index(z)

            val = sim.blobs[j,i]
            
            blob_vals.append(val)    

        if blob_vals:
            lp -= self.logprior_B(blob_vals)         

            # emcee will crash if this returns NaN
            if np.isnan(lp):
                return -np.inf, {}#self.blank_blob

        #if hasattr(sim, 'blobs'):
        #    blobs = sim.blobs
        #else:
        #    blobs = self.blank_blob    

        #if (not self.turning_points) and (not self.fit_signal):
        #    del sim, kw
        #    gc.collect()
        #    return lp, blobs

        # Compute the likelihood if we've made it this far
        if self.turning_points: 
                        
            try:
                nu = [nu_0_mhz / (1. + tps[tp][0]) \
                    for tp in self.turning_points]
                T = [tps[tp][1] for tp in self.turning_points]
            except KeyError:
                return -np.inf, self.blank_blob
            
            yarr = np.array(nu + T)
            
            assert len(yarr) == len(self.ydata)
                    
        else:
            yarr = np.interp(self.xdata, sim.history['nu'],            
                sim.history['dTb'])                                  
                
        if np.any(np.isnan(yarr)):
            return -np.inf, self.blank_blob
        
        like = 0.5 * (np.sum((yarr - self.ydata)**2 \
            / 2. / self.error**2 + np.log(2. * np.pi * self.error**2))) 
        logL = lp - like
        
        blobs = sim.blobs
                    
        del sim, kw
        gc.collect()
                    
        return logL, blobs

class FitGlobal21cm(ModelFit):
    def __init__(self, **kwargs):
        """
        Initialize a class for fitting the turning points in the global
        21-cm signal.

        Optional Keyword Arguments
        --------------------------
        Anything you want based to each ares.simulations.Global21cm call.
        
        """
        
        ModelFit.__init__(self, **kwargs)
        #self._prep_blobs()
        
    @property
    def loglikelihood(self):
        if not hasattr(self, '_loglikelihood'):
            self._loglikelihood = loglikelihood(self.xdata, self.ydata, 
                self.error, self.parameters, self.is_log, self.base_kwargs, 
                self.priors, self.prefix, self.blob_names, self.blob_ivars,
                self.blob_funcs, self.turning_points)    
        
        return self._loglikelihood
        
    @property
    def turning_points(self):
        if not hasattr(self, '_turning_points'):
            self._turning_points = list('BCD')
            if 'track_extrema' not in self.base_kwargs:
                print extrema_warning
                        
        return self._turning_points

    @turning_points.setter
    def turning_points(self, value):
        if type(value) == bool:
            if value:
                self._turning_points = list('BCD')
            else:
                self._turning_points = False
        elif type(value) == tuple:
            self._turning_points = list(value)
        elif type(value) == list:
            self._turning_points = value
        elif type(value) == str:
            if len(value) == 1:
                self._turning_points = [value]            
            else:
                self._turning_points = list(value)
        
        if self._turning_points is not None:
            if 'track_extrema' not in self.base_kwargs:
                print extrema_warning
            elif not self.base_kwargs['track_extrema']:
                print extrema_warning
                
    @property
    def data(self):
        if not hasattr(self, '_data'):
            raise AttributeError('Must set data by hand!')
        return self._data    
        
    @data.setter
    def data(self, value):
        if type(value) == dict:            
            kwargs = value.copy()
            kwargs.update(def_kwargs)
            
            sim = simGlobal21cm(**kwargs)
            sim.run()

            self.sim = sim

        elif isinstance(value, simGlobal21cm):
            sim = self.sim = value                   
        else:
            raise NotImplemented('help!')
            
        if self.turning_points:
            z = [sim.turning_points[tp][0] for tp in self.turning_points]
            T = [sim.turning_points[tp][1] for tp in self.turning_points]
                         
            nu = nu_0_mhz / (1. + np.array(z))
            ModelFit.xdata = nu
            ModelFit.ydata = np.array(T)
            
            self._data = np.array(list(nu) + T)    

    @property
    def error(self):
        if not hasattr(self, '_error'):
            raise AttributeError('Must set errors by hand!')
        return self._error
        
    @error.setter
    def error(self, value):
        if type(value) is dict:
                        
            nu = [value[tp][0] for tp in self.turning_points]
            T = [value[tp][1] for tp in self.turning_points]
            
            self._error = np.array(nu + T)
            
        else:
            if hasattr(self, '_data'):
                assert len(value) == len(self.data), \
                    "Data and errors must have same shape!"
                
            self._error = value
            
    #def _prep_blobs(self):
    #    """
    #    
    #    """                        
    #    if 'gaussian_model' in self.base_kwargs:
    #        if self.base_kwargs['gaussian_model']:
    #            return
    #    
    #    if 'inline_analysis' in self.base_kwargs:
    #        self.blob_names, self.blob_redshifts = \
    #            self.base_kwargs['inline_analysis']
    #    
    #    elif 'auto_generate_blobs' in self.base_kwargs:            
    #        if self.base_kwargs['auto_generate_blobs'] == True:
    #            kw = self.base_kwargs.copy()
    #
    #            sim = simG21(**kw)
    #            anl = InlineAnalysis(sim)
    #
    #            self.blob_names, self.blob_redshifts = \
    #                anl.generate_blobs()
    #            
    #            del sim, anl
    #            gc.collect()
    #            self.base_kwargs['inline_analysis'] = \
    #                (self.blob_names, self.blob_redshifts)
    #            self.base_kwargs['auto_generate_blobs'] = False
    #        elif self.base_kwargs['auto_generate_blobs'] == 'default':
    #            self.blob_names = _blob_names
    #            self.blob_redshifts = _blob_redshifts
    #    
    #    elif hasattr(self, 'blob_names'):
    #        self.base_kwargs['inline_analysis'] = \
    #            (self.blob_names, self.blob_redshifts)
    #    else:
    #        self.blob_names = self.blob_redshifts = None
    #
    #    if self.blob_redshifts is not None:
    #                
    #        TPs = 0
    #        for z in self.blob_redshifts:
    #            if z in list('BCD'):
    #                TPs += 1
    #        
    #        if self.turning_points:  
    #            msg = 'Fitting won\'t work if no turning points are provided.'
    #            assert TPs > 0, msg
    #        
    #    self.one_file_per_blob = self.base_kwargs['one_file_per_blob']        
    #        
    def _check_for_conflicts(self):
        """
        Hacky at the moment. Preventative measure against is_log=True for
        spectrum_logN. Could generalize.
        """
        for i, element in enumerate(self.parameters):
            if re.search('spectrum_logN', element):
                if self.is_log[i]:
                    raise ValueError('spectrum_logN is already logarithmic!')


