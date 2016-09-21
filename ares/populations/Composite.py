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
from .GalaxyCohort import GalaxyCohort
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
        to_quantity = [None for i in range(self.Npops)]
        for i, pf in enumerate(self.pfs):
                        
            if re.search('link', pf['pop_sfr_model']):
                junk, linkto, linkee = pf['pop_sfr_model'].split(':')
                to_tunnel[i] = int(linkee)
                to_quantity[i] = linkto
            else:
                self.pops[i] = GalaxyPopulation(**pf)

        # Establish a link from one population's attribute to another
        for i, entry in enumerate(to_tunnel):
            if entry is None:
                continue
                        
            tmp = self.pfs[i].copy()
            
            if to_quantity[i] == 'sfrd':
                self.pops[i] = GalaxyAggregate(**tmp)
                self.pops[i]._sfrd = self.pops[entry]._sfrd_func
            elif to_quantity[i] in ['sfe', 'fstar']:
                self.pops[i] = GalaxyCohort(**tmp)
                self.pops[i]._fstar = self.pops[entry].SFE
            else:
                raise NotImplementedError('help')

        # Set ID numbers (mostly for debugging purposes)
        for i, pop in enumerate(self.pops):
            pop.id_num = i
            


