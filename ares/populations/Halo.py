"""

Halo.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu May 28 16:22:44 MDT 2015

Description: 

"""

import numpy as np
from .Population import Population
from ..physics import HaloMassFunction
from ..util.PrintInfo import print_pop

class HaloPopulation(Population):
    def __init__(self, **kwargs):
        
        # This is basically just initializing an instance of the cosmology
        # class. Also creates the parameter file attribute ``pf``.
        Population.__init__(self, **kwargs)

        # Print info to screen
        if self.pf['verbose']:
            self.info

        # Setup splines for interpolation of dfcoll/dt
        self._init_pop()

    @property
    def parameterized(self):
        if not hasattr(self, '_parameterized'):
            not_parameterized = (self.pf['pop_k_ion_igm']) is None
            not_parameterized &= (self.pf['pop_k_ion_cgm']) is None
            not_parameterized &= (self.pf['pop_k_heat_igm']) is None
            self._parameterized = not not_parameterized

        return self._parameterized

    @property
    def info(self):
        if not self.parameterized:
            print_pop(self)

    @property
    def fcoll(self):
        if not hasattr(self, '_fcoll'):
            self._init_pop()
    
        return self._fcoll
    
    @property
    def dfcolldz(self):
        if not hasattr(self, '_dfcolldz'):
            self._init_pop()
    
        return self._dfcolldz
        
    def dfcolldt(self, z):
        return self.dfcolldz(z) / self.cosm.dtdz(z)    
    
    def _set_fcoll(self, Tmin, mu):
        self._fcoll, self._dfcolldz, self._d2fcolldz2 = \
            self.halos.build_1d_splines(Tmin, mu)

    @property
    def halos(self):
        if not hasattr(self, '_halos'):
            self._halos = HaloMassFunction(**self.pf)
        return self._halos

    def _init_pop(self):
        # Halo stuff
        if self.pf['pop_sfrd'] is not None:
            return

        if self.pf['pop_fcoll'] is None:
            self._set_fcoll(self.pf['pop_Tmin'], self.pf['mu'])
        else:
            self._fcoll, self._dfcolldz = \
                self.pf['pop_fcoll'], self.pf['pop_dfcolldz']
    
        