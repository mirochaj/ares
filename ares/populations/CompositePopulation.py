""" 
CompositePopulation.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2012-02-17.

Description: Define a population of objects - return list of Population____
class instances.

"""

import re
import numpy as np
from ..util.Misc import parse_kwargs
from .StellarPopulation import StellarPopulation
from .BlackHolePopulation import BlackHolePopulation
            
class CompositePopulation:
    """
    Initialize a CompositePopulation object, i.e., a list of *Population
    instances.
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs.copy()
        self.other_kw = self.kwargs.copy()
        
        N = 1
        # Determine number of populations
        for par in ['source_kwargs', 'spectrum_kwargs']:
            if par in self.other_kw:
                self.other_kw.pop(par)
            
            if par not in self.kwargs:
                continue
            if self.kwargs[par] is None:
                continue    
            
            if type(self.kwargs[par]) is list:    
                N = max(N, len(self.kwargs[par]))                    
                
        self.N = N        
        self.pfs = [self.other_kw.copy() for i in xrange(N)]    
        
        # If source_kwargs/spectrum_kwargs supplied, make list of parameter files 
        for i, par in enumerate(['source_kwargs', 'spectrum_kwargs']):
            if par not in kwargs:
                continue
                
            kw = kwargs[par] # source_kwargs or spectrum_kwargs
            
            # Add components to each population's pf
            if type(kw) is list and N > 1:
                for j in xrange(N):
                    self.pfs[j].update(kw[j])
            elif type(kw) is dict:
                for j in xrange(N):
                    self.pfs[j].update(kw)
                        
        self.BuildPopulationInstances()
        
    def BuildPopulationInstances(self):
        """
        Construct list of ____Population class instances.
        """
        
        self.pops = []
        for pf in self.pfs:  # List of dictionaries, one for each pop
            if pf['source_type'] == 'star':
                self.pops.append(StellarPopulation(**pf))
            elif pf['source_type'] == 'bh':
                self.pops.append(BlackHolePopulation(**pf))    
            else:
                raise ValueError('Unrecognized source_type %s.' % pf['source_type'])  


