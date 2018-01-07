"""

FitMulti.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Sat Jan  6 20:41:18 PST 2018

Description: 

"""

from __future__ import print_function

import gc
import time
import numpy as np
from .ModelFit import ModelFit
from ..util.Pickling import read_pickle_file, write_pickle_file

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1
    
class FitMulti(ModelFit): 
    
    @property
    def simulator(self):
        if not hasattr(self, '_simulator'):
            raise AttributeError('Must supply simulator by hand!')
        return self._simulator
        
    @simulator.setter
    def simulator(self, value):
        self._simulator = value
        
        # Assert that this is an instance of something we know about?

    @property
    def fitters(self):
        return self._fitters
    
    def add_fitter(self, fitter):
        if not hasattr(self, '_fitters'):
            self._fitters = []
            
        if fitter in self._fitters:
            print("This fitter is already included!")
            return
            
        self._fitters.append(fitter)
    
    def loglikelihood(self, pars):
        
        kwargs = {}
        for i, par in enumerate(self.parameters):
        
            if self.is_log[i]:
                kwargs[par] = 10**pars[i]
            else:
                kwargs[par] = pars[i]

        # Apply prior on model parameters first (dont need to generate model)
        point = {}
        for i in range(len(self.parameters)):
            point[self.parameters[i]] = pars[i]
            
        lp = self.prior_set_P.log_value(point)
                
        if not np.isfinite(lp):
            return -np.inf, self.blank_blob

        # Update kwargs
        kw = self.base_kwargs.copy()
        kw.update(kwargs)

        # Don't save base_kwargs for each proc! Needlessly expensive I/O-wise.
        self.checkpoint(**kwargs)

        t1 = time.time()
        sim = self.sim = self.simulator(**kw)
        sim.run()
        t2 = time.time()
        
        lnL = 0.0
        for fitter in self.fitters:
            lnL += fitter.loglikelihood(sim)
                    
        # Final posterior calculation
        PofD = lp - lnL
                
        if np.isnan(PofD):
            return -np.inf, self.blank_blob
            
        try:
            blobs = sim.blobs
        except:
            blobs = self.blank_blob
                    
        del sim, kw
        gc.collect()    

        return PofD, blobs        

    def _compute_blob_prior(self, sim):
        blob_vals = {}
        for key in self.priors_B.params:
    
            grp, i, nd, dims = sim.blob_info(key)
    
            #if nd == 0:
            #    blob_vals[key] = sim.get_blob(key)
            #elif nd == 1:    
            blob_vals[key] = sim.get_blob(key)
            #else:
            #    raise NotImplementedError('help')
    
        try:
            # will return 0 if there are no blobs
            return self.priors_B.log_value(blob_vals)
        except:
            # some of the blobs were not retrieved (then they are Nones)!
            return -np.inf
    
    @property
    def blank_blob(self):
        if not hasattr(self, '_blank_blob'):
    
            if self.blob_names is None:
                self._blank_blob = []
                return []
    
            self._blank_blob = []
            for i, group in enumerate(self.blob_names):
                if self.blob_ivars[i] is None:
                    self._blank_blob.append([np.inf] * len(group))
                else:
                    if self.blob_nd[i] == 0:
                        self._blank_blob.append([np.inf] * len(group))
                    elif self.blob_nd[i] == 1:
                        arr = np.ones([len(group), self.blob_dims[i][0]])
                        self._blank_blob.append(arr * np.inf)
                    elif self.blob_nd[i] == 2:
                        dims = len(group), self.blob_dims[i][0], \
                            self.blob_dims[i][1]
                        arr = np.ones(dims)
                        self._blank_blob.append(arr * np.inf)
    
        return self._blank_blob
    
    def checkpoint(self, **kwargs):
        if self.checkpoint_by_proc:
            procid = str(rank).zfill(3)
            fn = '{0!s}.{1!s}.checkpt.pkl'.format(self.prefix, procid)
            write_pickle_file(kwargs, fn, ndumps=1, open_mode='w',\
                safe_mode=False, verbose=False)
    
            fn = '{0!s}.{1!s}.checkpt.txt'.format(self.prefix, procid)
            with open(fn, 'w') as f:
                print("Simulation began: {!s}".format(time.ctime()), file=f)
    
    def checkpoint_on_completion(self, **kwargs):
        if self.checkpoint_by_proc:
            procid = str(rank).zfill(3)
            fn = '{0!s}.{1!s}.checkpt.txt'.format(self.prefix, procid)
            with open(fn, 'a') as f:
                print("Simulation finished: {!s}".format(time.ctime()), file=f)
    
    
    