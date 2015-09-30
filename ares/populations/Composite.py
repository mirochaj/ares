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
from ..util import ParameterFile
from .Galaxy import GalaxyPopulation
            
class CompositePopulation:
    def __init__(self, **kwargs):
        """
        Initialize a CompositePopulation object, i.e., a list of *Population instances.
        """
        
        self.pf = ParameterFile(**kwargs)
        
        N = self.Npops = self.pf.Npops
        self.pfs = self.pf.pfs
                    
        self.BuildPopulationInstances()
        
    def BuildPopulationInstances(self):
        """
        Construct list of *Population class instances.
        """
        
        self.pops = []
        for i, pf in enumerate(self.pfs):
            if pf['pop_type'] == 'galaxy':
                self.pops.append(GalaxyPopulation(**pf))
            else:
                raise ValueError('Unrecognized pop_type %s.' % pf['pop_type'])  


