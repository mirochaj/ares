"""

ModelFit.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon May 12 14:01:29 MDT 2014

Description: 

"""

import signal
import numpy as np
from ..util.Pickling import write_pickle_file
from ..physics.Constants import nu_0_mhz
import gc, os, sys, copy, types, time, re
from .ModelFit import ModelFit, LogLikelihood, FitBase
from ..simulations import Global21cm as simG21
from ..analysis import Global21cm as anlGlobal21cm
from ..simulations import Global21cm as simGlobal21cm
try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str
    
try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1
     
def_kwargs = {'verbose': False, 'progress_bar': False}

class loglikelihood(LogLikelihood):
    def __init__(self, xdata, ydata, error, turning_points):
        """
        Computes log-likelihood at given step in MCMC chain.
    
        Parameters
        ----------
    
        """
        
        LogLikelihood.__init__(self, xdata, ydata, error)
        self.turning_points = turning_points
        
    def __call__(self, sim):
        """
        Compute log-likelihood for model generated via input parameters.

        Returns
        -------
        Tuple: (log likelihood, blobs)

        """
                                                        
        # Compute the likelihood if we've made it this far
        if self.turning_points:
            tps = sim.turning_points
            
            try:
                nu = [nu_0_mhz / (1. + tps[tp][0]) \
                    for tp in self.turning_points]
                T = [tps[tp][1] for tp in self.turning_points]
            except KeyError:
                return -np.inf

            yarr = np.array(nu + T)

            assert len(yarr) == len(self.ydata)

        else:
            yarr = np.interp(self.xdata, sim.history['nu'], sim.history['dTb'])

        if np.any(np.isnan(yarr)):
            return -np.inf

        lnL = -0.5 * (np.sum((yarr - self.ydata)**2 \
            / self.error**2 + np.log(2. * np.pi * self.error**2)))
                                                
        return lnL + self.const_term

class FitGlobal21cm(FitBase):

    @property
    def loglikelihood(self):
        if not hasattr(self, '_loglikelihood'):
            self._loglikelihood = loglikelihood(self.xdata, self.ydata, 
                self.error, self.turning_points)
        
        return self._loglikelihood
        
    @property
    def turning_points(self):
        if not hasattr(self, '_turning_points'):
            self._turning_points = False

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
        elif isinstance(value, basestring):
            if len(value) == 1:
                self._turning_points = [value]            
            else:
                self._turning_points = list(value)
                
    @property
    def frequencies(self):
        if not hasattr(self, '_frequencies'):
            raise AttributeError('Must supply frequencies by hand!')
        return self._frequencies
        
    @frequencies.setter
    def frequencies(self, value):
        self._frequencies = value
                        
    @property
    def data(self):
        if not hasattr(self, '_data'):
            raise AttributeError('Must set data by hand!')
        return self._data    
        
    @data.setter
    def data(self, value):
        """
        Set x and ydata at the same time, either by passing in 
        a simulation instance, a dictionary of parameters, or a 
        sequence of brightness temperatures corresponding to the
        frequencies defined in self.frequencies (self.xdata).
        """
                
        if type(value) == dict:            
            kwargs = value.copy()
            kwargs.update(def_kwargs)

            sim = simGlobal21cm(**kwargs)
            sim.run()

            self.sim = sim

        elif isinstance(value, simGlobal21cm) or \
             isinstance(value, anlGlobal21cm):
            sim = self.sim = value
        elif type(value) in [list, tuple]:
            sim = None
        else:
            assert len(value) == len(self.frequencies)            
            assert not self.turning_points
            self.xdata = self.frequencies
            self.ydata = value
            
            return

        if self.turning_points is not None:
            
            self.xdata = None
            if sim is not None:
                z = [sim.turning_points[tp][0] for tp in self.turning_points]
                T = [sim.turning_points[tp][1] for tp in self.turning_points]
                
                nu = nu_0_mhz / (1. + np.array(z))
                
                self.ydata = np.array(list(nu) + T)
            else:
                assert len(value) == 2 * len(self.turning_points)
                self.ydata = value
                
        else:
            
            self.xdata = self.frequencies
            if hasattr(self, 'sim'):
                nu = self.sim.history['nu']
                dTb = self.sim.history['dTb']
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


