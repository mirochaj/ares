"""

MetaGalacticBackground.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Feb 16 12:43:06 MST 2015

Description: 

"""

import numpy as np
from ..util import ParameterFile
from ..static import GlobalVolume
from ..solvers import UniformBackground
from ..populations import CompositePopulation

igm_pars = \
{
 'grid_cells': 1,
 'isothermal': False,
 'expansion': True,
 'initial_ionization': [1.-1.2e-3, 1.2e-3],
 'cosmological_ics': True,
}

bands = ['lw', 'uv', 'xr']

class MetaGalacticBackground:
    def __init__(self, **kwargs):
        """
        Initialize a MetaGalacticBackground object.
        
        .. note:: This class assumes an optically thin Universe at all 
            photon energies. For self-consistent solutions including an 
            evolving IGM opacity, check out the MultiPhaseIGM class.
            
        """
        self.pf = ParameterFile(**kwargs)

        self._set_radiation_field()
        
    def _set_radiation_field(self):
        """
        Loop over populations, make separate RB and RS instances for each.
        """
    
        self.field = UniformBackground(grid=None, **self.pf)
   
    def run(self, t, dt):
        """
        Evolve radiation background in time.

        .. note:: Assumes we're using the generator, otherwise the time 
            evolution must be controlled manually.
            
        """
        
        pass
        
    def step(self):
        pass
        
        
        

