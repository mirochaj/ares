"""

ModelSample.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Wed Jul 13 11:14:11 PDT 2016

Description: 

"""

import numpy as np
from .ModelGrid import ModelGrid

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
    def axes(self):
        return self.prior_set.params
    
    @property
    def N(self):
        return self._N
    @N.setter
    def N(self, value):
        self._N = int(value)
        
    def run(self, prefix, clobber=False, restart=False, save_freq=500):
        
        # Initialize space
        models = []
        for i in range(self.N):
            kw = self.prior_set.draw()
            models.append(kw)
            
        self.set_models(models)
        
        super(ModelSample, self).run(prefix, clobber, restart, save_freq)
        