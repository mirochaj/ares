"""

TwoZoneIGM.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Feb 16 12:46:28 MST 2015

Description: 

"""

import numpy as np
from .GasParcel import GasParcel
from ..util import ParameterFile
from .MetaGalacticBackground import MetaGalacticBackground

igm_pars = \
{
 'grid_cells': 1,
 'isothermal': False,
 'initial_ionization': [1. - 1.2e-3, 1.2e-3],
 'initial_temperature': 1e2,
 'expansion': True,
 'cosmological_ics': True,
}

cgm_pars = \
{
 'grid_cells': 1,
 'isothermal': True,
 'initial_ionization': [1e-8, 1. - 1e-8],
 'intial_temperature': 1e4,
 'expansion': True,
 'cosmological_ics': False,
}

class TwoZoneIGM:
    def __init__(self, **kwargs):
        
        # Initialize two GasParcels
        kw_igm = kwargs.copy()
        kw_igm.update(igm_pars)
        
        kw_cgm = kwargs.copy()
        kw_cgm.update(cgm_pars)        
        
        self.grid_igm = GasParcel(**kw_igm)
        self.grid_cgm = GasParcel(**kw_cgm)
    
        # Initialize generators
        self.gen_igm = self.grid_igm.step()
        self.gen_cgm = self.grid_cgm.step()
        
        # Initialize radiation backgrounds?
        
    def evolve(self, t, dt):
        pass
        
    def step(self, t, dt):
        pass    
    
    def tau(self):
        pass
        
        

        