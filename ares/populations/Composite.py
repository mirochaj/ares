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
try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

after_instance = ['pop_rad_yield']
allowed_options = ['pop_sfr_model', 'pop_Mmin']

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
        to_copy = [None for i in range(self.Npops)]
        to_attribute = [None for i in range(self.Npops)]
        for i, pf in enumerate(self.pfs):
                        
            ct = 0            
            # Only link options that are OK at this stage.
            for option in allowed_options:
                                
                if (pf[option] is None) or (not isinstance(pf[option], basestring)):
                    # Only can happen for pop_Mmin
                    continue
                                
                if re.search('link', pf[option]):
                    try:
                        junk, linkto, linkee = pf[option].split(':')
                        to_tunnel[i] = int(linkee)
                        to_quantity[i] = linkto
                    except ValueError:
                        # Backward compatibility issue: we used to only ever
                        # link to the SFRD of another population
                        junk, linkee = pf[option].split(':')
                        to_tunnel[i] = int(linkee)
                        to_quantity[i] = 'sfrd'
                        assert option == 'pop_sfr_model'
                        print('HELLO help please')
                        
                    ct += 1    
            
            assert ct < 2
            
            if ct == 0:
                self.pops[i] = GalaxyPopulation(**pf)
            
            # This is poor design, but things are setup such that only one
            # quantity can be linked. This is a way around that.
            for option in after_instance:
                if (pf[option] is None) or (not isinstance(pf[option], basestring)):
                    # Only can happen for pop_Mmin
                    continue
                
                if re.search('link', pf[option]):
                    junk, linkto, linkee = pf[option].split(':')
                    to_copy[i] = int(linkee)
                    to_attribute[i] = linkto

        # Establish a link from one population's attribute to another
        for i, entry in enumerate(to_tunnel):
            if entry is None:
                continue
                        
            tmp = self.pfs[i].copy()
            
            if self.pops[i] is not None:
                raise ValueError('Only one link allowed right now!')
                
            if to_quantity[i] in ['sfrd', 'emissivity']:
                self.pops[i] = GalaxyAggregate(**tmp)
                self.pops[i]._sfrd = self.pops[entry]._sfrd_func
            elif to_quantity[i] in ['sfe', 'fstar']:
                self.pops[i] = GalaxyCohort(**tmp)
                self.pops[i]._fstar = self.pops[entry].SFE
            elif to_quantity[i] in ['Mmax_active']:
                self.pops[i] = GalaxyCohort(**tmp)
                self.pops[i]._tab_Mmin = self.pops[entry]._tab_Mmax_active
            elif to_quantity[i] in ['Mmax']:
                self.pops[i] = GalaxyCohort(**tmp)
                # You'll notice that we're assigning what appears to be an 
                # array to something that is a function. Fear not! The setter
                # for _tab_Mmin will sort this out.
                self.pops[i]._tab_Mmin = self.pops[entry].Mmax
                assert np.all(self.pops[i]._tab_Mmin <= self.pops[entry]._tab_Mmax)
            elif to_quantity[i] in after_instance:
                continue
            else:
                raise NotImplementedError('help')

        # Set ID numbers (mostly for debugging purposes)
        for i, pop in enumerate(self.pops):
            pop.id_num = i

        # Posslible few last things that occur after Population objects made
        for i, entry in enumerate(to_copy):
            if entry is None:
                continue

            tmp = self.pfs[i].copy()

            self.pops[i].yield_per_sfr = \
                self.pops[entry].__getattribute__(to_attribute[i])
        
        
                        
