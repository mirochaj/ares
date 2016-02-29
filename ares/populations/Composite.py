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
from .GalaxyAggregate import GalaxyAggregate
from .GalaxyPopulation import GalaxyPopulation

class CompositePopulation(object):
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
                if pf['pop_model'] == 'fcoll':
                    self.pops[i] = GalaxyAggregate(**pf)
                else:
                    self.pops[i] = GalaxyPopulation(**pf)
            else:
                raise ValueError('Unrecognized pop_type %s.' % pf['pop_type'])  

        # Tunneling populations!
        for i, entry in enumerate(to_tunnel):
            if self.pfs[i]['pop_tunnel'] is None:
                continue
                        
            tmp = self.pfs[i].copy()
            
            # This is the tunnel
            tmp['pop_sfrd'] = self.pops[entry]._sfrd_func

            # Only makes sense to tunnel to non-fcoll model
            self.pops[i] = GalaxyAggregate(**tmp)

            
            


