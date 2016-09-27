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
#from .Parameterized import ParametricPopulation


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
                        
            if re.search('link', pf['pop_sfr_model']):
                junk, linkto = pf['pop_sfr_model'].split(':')
                to_tunnel[i] = int(linkto)
            else:
                self.pops[i] = GalaxyPopulation(**pf)

        # Tunneling populations!
        for i, entry in enumerate(to_tunnel):
            if entry is None:
                continue
                        
            tmp = self.pfs[i].copy()
            
            # This is the tunnel
            #tmp['pop_sfrd'] = self.pops[entry]._sfrd_func

            # Only makes sense to tunnel to non-fcoll model
            self.pops[i] = GalaxyAggregate(**tmp)
            self.pops[i]._sfrd = self.pops[entry]._sfrd_func

        # Set ID numbers (mostly for debugging purposes)
        for i, pop in enumerate(self.pops):
            pop.id_num = i
            


