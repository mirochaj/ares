"""

ModelFit.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon May 12 14:01:29 MDT 2014

Description: 

"""

import numpy as np
from ..util.PrintInfo import print_fit
from ..physics.Constants import nu_0_mhz
import gc, os, sys, copy, types, time, re
from .ModelFit import ModelFit, LogLikelihood
from ..simulations import Global21cm as simG21
from ..analysis import Global21cm as anlGlobal21cm
from ..analysis.InlineAnalysis import InlineAnalysis
from ..simulations import Global21cm as simGlobal21cm
 
def_kwargs = {'verbose': False, 'progress_bar': False}

class loglikelihood(LogLikelihood):
    def __init__(self, xdata, ydata, error, parameters, is_log,
        base_kwargs, param_prior_set=None, blob_prior_set=None, 
        prefix=None, blob_info=None, turning_points=None):
        """
        Computes log-likelihood at given step in MCMC chain.

        Parameters
        ----------

        """
        
        LogLikelihood.__init__(self, xdata, ydata, error, parameters, is_log,
            base_kwargs, param_prior_set, blob_prior_set, 
            prefix, blob_info)

        self.turning_points = turning_points
        
        if self.turning_points is not None:

            nu = [xdata[i] for i, tp in enumerate(self.turning_points)]
            T = [ydata[i] for i, tp in enumerate(self.turning_points)]
            
            self.xdata = None
            self.ydata = np.array(nu + T)
        else:
            self.xdata = xdata
            self.ydata = ydata
            
        self.error = error
        
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
        
        point = {}
        for i in range(len(self.parameters)):
            point[self.parameters[i]] = pars[i]
        
        lp = self.priors_P.log_prior(point)
        if not np.isfinite(lp):
            return -np.inf, self.blank_blob

        # Run a model and retrieve turning points
        kw = self.base_kwargs.copy()
        kw.update(kwargs)

        try:
            sim = self.sim = simG21(**kw)
            sim.run()
                        
            tps = sim.turning_points
                                        
        # Timestep weird (happens when xi ~ 1)
        except SystemExit:
            
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

        if self.priors_B.params != []:
            lp += self._compute_blob_prior(sim)
        
        # emcee will crash if this returns NaN
        if np.isnan(lp):
            return -np.inf, self.blank_blob

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
            yarr = np.interp(self.xdata, sim.data['nu'],            
                sim.history['igm_dTb'])                                  
                
        if np.any(np.isnan(yarr)):
            return -np.inf, self.blank_blob
        
        like = 0.5 * (np.sum((yarr - self.ydata)**2 \
            / self.error**2 + np.log(2. * np.pi * self.error**2))) 
        logL = lp - like
                
        blobs = sim.blobs
                    
        del sim, kw
        gc.collect()
                    
        return logL, blobs

class FitGlobal21cm(ModelFit):
        
    @property
    def loglikelihood(self):
        if not hasattr(self, '_loglikelihood'):
            self._loglikelihood = loglikelihood(self.xdata, self.ydata, 
                self.error, self.parameters, self.is_log, self.base_kwargs, 
                self.prior_set_P, self.prior_set_B, 
                self.prefix, self.blob_info, self.turning_points)    
        
        return self._loglikelihood
        
    @property
    def turning_points(self):
        if not hasattr(self, '_turning_points'):
            self._turning_points = None

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
            assert len(value) == len(self.frequencies)            
            ModelFit.ydata = value

        if self.turning_points:
            z = [sim.turning_points[tp][0] for tp in self.turning_points]
            T = [sim.turning_points[tp][1] for tp in self.turning_points]

            nu = nu_0_mhz / (1. + np.array(z))
            self.xdata = nu
            self.ydata = np.array(T)

            self._data = np.array(list(nu) + T)
        else:
            assert self.frequencies is not None, \
                "Must set frequencies by hand or set turning_points."
            
            self.xdata = self.frequencies
            if hasattr(self, 'sim'):
                nu = self.sim.data['nu']
                dTb = self.sim.data['igm_dTb']
                self.ydata = np.interp(self.xdata, nu, dTb).copy() \
                    + self.noise

    @property
    def noise(self):
        if not hasattr(self, '_noise'):
            self._noise = np.zeros_like(self.xdata)
        return self._noise
    
    @noise.setter
    def noise(self, value):
        self._noise = np.random.normal(0., value, size=len(self.frequencies))
            
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
          
    def _check_for_conflicts(self):
        """
        Hacky at the moment. Preventative measure against is_log=True for
        spectrum_logN. Could generalize.
        """
        for i, element in enumerate(self.parameters):
            if re.search('spectrum_logN', element):
                if self.is_log[i]:
                    raise ValueError('spectrum_logN is already logarithmic!')


