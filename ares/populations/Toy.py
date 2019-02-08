"""

ToyPopulation.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Tue Apr 17 12:29:50 PDT 2018

Description: 

"""

import numpy as np
from ..physics import Cosmology
from .Population import Population
from ares.physics.Constants import c, erg_per_ev

class Toy(Population):
    
    def Emissivity(self, z, E=None, Emin=None, Emax=None):
        return np.zeros_like(z)
        
        