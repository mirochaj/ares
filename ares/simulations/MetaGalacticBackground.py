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
        """
        self.pf = ParameterFile(**kwargs)
        self.pf.update(igm_pars)
                
        self._set_sources()
        self._set_radiation_field()
        
    def _set_sources(self):            
        """
        Initialize radiation sources and radiative transfer solver.
        """

        self.sources = CompositePopulation(**self.pf)

        # Determine if backgrounds are approximate or not
        self.approx_all_xrb = 1
        self.approx_all_lwb = 1
        for pop in self.sources.pops:
            self.approx_all_xrb *= pop.pf['approx_xrb']
            self.approx_all_lwb *= pop.pf['approx_lwb']
        
    def _set_radiation_field(self):
        """
        Loop over populations, make separate RB and RS instances for each.
        """
        
        if self.approx_all_xrb * self.approx_all_lwb:
            return
        
        self.Nrbs = self.sources.Npops
        self.field = [UniformBackground(pop) for pop in self.sources.pops]

        self.all_discrete_lwb = 1
        self.all_discrete_xrb = 1
        for field in self.field:
            self.all_discrete_lwb *= field.pf['is_lya_src']
            self.all_discrete_xrb *= field.pf['is_heat_src_igm']

    def run(self, t, dt):
        """
        Evolve radiation background in time.

        .. note:: Assumes we're using the generator, otherwise the time 
            evolution must be controlled manually.
            
        """
        
        pass
        
    def step(self):
        pass
        
        
        

