"""

ModelSample.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Wed Jul 13 11:14:11 PDT 2016

Description: 

"""

import numpy as np
from .ModelGrid import ModelGrid

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1

class ModelSample(ModelGrid):
    
    @property
    def prior_set(self):
        return self._prior_set
    
    @prior_set.setter
    def prior_set(self, value):
        self._prior_set = value

    @property
    def seed(self):
        if hasattr(self, '_seed'):
            return self._seed
        return None

    @seed.setter
    def seed(self, value):
        self._seed = value

    @property            
    def parameters(self):
        if not hasattr(self, '_parameters'):
            self._parameters = self.prior_set.params
        return self._parameters

    @property
    def N(self):
        return self._N
    @N.setter
    def N(self, value):
        self._N = int(value)
    
    @property
    def is_log(self):
        if not hasattr(self, '_is_log'):
            self._is_log = [False] * len(self.parameters)
        return self._is_log
        
    @is_log.setter
    def is_log(self, value):
        if type(value) in [int, bool]:
            self._is_log = value
        elif type(value) is dict:
            self._is_log = []
            for par in self.parameters:
                if par in value:
                    self._is_log.append(value[par])
                else:
                    self._is_log.append(False)
        else:
            assert len(value) == len(self.parameters)
            self._is_log = value
        
    def get_models(self):
        if rank == 0:

            np.random.seed(self.seed)

            models = []
            for i in range(self.N):
                kw = self.prior_set.draw()
                models.append(kw)

        else:
            models = None

        if size > 1:
            models = MPI.COMM_WORLD.bcast(models, root=0)

        return models

    def run(self, prefix, clobber=False, restart=False, save_freq=500):
        """
        Run self.N models.

        Parameters
        ----------
        prefix : str
            Prefix for all output files.
        save_freq : int
            Number of steps to take before writing data to disk. Note that if
            you're running in parallel, this is the number of steps *each 
            processor* will take before writing to disk.
        clobber : bool
            Overwrite pre-existing files of the same prefix if one exists?
        restart : bool
            Append to pre-existing files of the same prefix if one exists?

        Returns
        -------
        
        """
        
        models = self.get_models()
            
        # Initialize space -- careful if running in parallel
        self.set_models(models)
                
        super(ModelSample, self).run(prefix, clobber, restart, save_freq)
        