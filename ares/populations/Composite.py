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
        
        self.pops = [None for i in range(self.Npops)]
        to_tunnel = [None for i in range(self.Npops)]
        for i, pf in enumerate(self.pfs):
            
            if pf['pop_tunnel'] is not None:
                to_tunnel[i] = pf['pop_tunnel']
            elif pf['pop_type'] == 'galaxy':
                self.pops[i] = GalaxyPopulation(**pf)
            else:
                raise ValueError('Unrecognized pop_type %s.' % pf['pop_type'])  

        # Tunneling populations!
        for i, entry in enumerate(to_tunnel):
            if self.pfs[i]['pop_tunnel'] is None:
                continue
            
            tmp = self.pfs[i].copy()
            tmp['pop_sfrd'] = self.pops[to_tunnel[i]].SFRD
            
            self.pops[i] = GalaxyPopulation(**tmp)
            
            
            


