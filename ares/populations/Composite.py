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
from ..util.Misc import get_attribute
from .GalaxyCohort import GalaxyCohort
from .GalaxyAggregate import GalaxyAggregate
from .GalaxyPopulation import GalaxyPopulation
from .BlackHoleAggregate import BlackHoleAggregate

try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

after_instance = ['pop_rad_yield']
allowed_options = ['pop_sfr_model', 'pop_Mmin', 'pop_frd']

class CompositePopulation(object):
    def __init__(self, pf=None, cosm=None, **kwargs):
        """
        Initialize a CompositePopulation object, i.e., a list of *Population instances.
        """

        if pf is None:
            self.pf = ParameterFile(**kwargs)
        else:
            self.pf = pf

        N = self.Npops = self.pf.Npops
        self.pfs = self.pf.pfs
        self._cosm_ = cosm

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
        link_args = [[] for i in range(self.Npops)]
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
                self.pops[i] = GalaxyPopulation(cosm=self._cosm_, **pf)

            # This is poor design, but things are setup such that only one
            # quantity can be linked. This is a way around that.
            for option in after_instance:
                if (pf[option] is None) or (not isinstance(pf[option], basestring)):
                    # Only can happen for pop_Mmin
                    continue

                if re.search('link', pf[option]):
                    options = pf[option].split(':')

                    if len(options) == 3:
                        junk, linkto, linkee = options
                        args = None
                    elif len(options) == 4:
                        junk, linkto, linkee, args = options
                    else:
                        raise ValueError('Wrong number of options supplied via link!')

                    to_copy[i] = int(linkee)
                    to_attribute[i] = linkto

                    if args is not None:
                        link_args[i] = map(float, args.split('-'))

        # Establish a link from one population's attribute to another
        for i, entry in enumerate(to_tunnel):
            if entry is None:
                continue

            tmp = self.pfs[i].copy()

            if self.pops[i] is not None:
                raise ValueError('Only one link allowed right now!')

            if to_quantity[i] in ['sfrd', 'emissivity']:
                self.pops[i] = GalaxyAggregate(cosm=self._cosm_, **tmp)
                self.pops[i]._sfrd = self.pops[entry]._sfrd_func
            elif to_quantity[i] in ['frd']:
                self.pops[i] = BlackHoleAggregate(cosm=self._cosm_, **tmp)
                self.pops[i]._frd = self.pops[entry]._frd_func
            elif to_quantity[i] in ['sfe', 'fstar']:
                self.pops[i] = GalaxyCohort(cosm=self._cosm_, **tmp)
                self.pops[i]._fstar = self.pops[entry].SFE
            elif to_quantity[i] in ['Mmax_active']:
                self.pops[i] = GalaxyCohort(cosm=self._cosm_, **tmp)
                self.pops[i]._tab_Mmin = self.pops[entry]._tab_Mmax_active
            elif to_quantity[i] in ['Mmax']:
                self.pops[i] = GalaxyCohort(cosm=self._cosm_, **tmp)
                # You'll notice that we're assigning what appears to be an
                # array to something that is a function. Fear not! The setter
                # for _tab_Mmin will sort this out.
                self.pops[i]._tab_Mmin = self.pops[entry].Mmax

                ok = self.pops[i]._tab_Mmin <= self.pops[entry]._tab_Mmax
                excess = self.pops[i]._tab_Mmin - self.pops[entry]._tab_Mmax

                # For some reason there's a machine-dependent tolerance issue
                # here that causes a crash in a hard-to-reproduce way.
                if not np.all(ok):
                    err_str = "{}/{} elements not abiding by condition.".format(
                            ok.size - ok.sum(), ok.size)
                    err_str += " Typical (Mmin - Mmax) = {}".format(np.mean(excess[~ok]))

                    if excess[~ok].mean() < 1e-4:
                        pass
                    else:
                        assert np.all(ok), err_str

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

            args = link_args[i]

            # If the attribute is just an attribute (i.e., no nesting)
            if '.' not in to_attribute[i]:
                self.pops[i].yield_per_sfr = \
                    self.pops[entry].__getattribute__(to_attribute[i])

                continue

            ##
            # Nested attributes
            ##
    
            # Recursively find the attribute we want
            func = get_attribute(to_attribute[i], self.pops[entry])

            # This may need to be generalized if the nested attribute
            # is not a function.
            self.pops[i].yield_per_sfr = func(*args)
