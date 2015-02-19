"""

MetaGalacticBackground.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Feb 16 12:43:06 MST 2015

Description: 

"""

import numpy as np
from ..util import ParameterFile
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
        
    def _set_sources(self):            
        """
        Initialize radiation sources and radiative transfer solver.
        """
    
        self.pops = CompositePopulation(**self.pf)
    
        # Determine if backgrounds are approximate or not
        self.approx_all_xrb = 1
        self.approx_all_lwb = 1
        for pop in self.pops.pops:
            self.approx_all_xrb *= pop.pf['approx_xrb']
            self.approx_all_lwb *= pop.pf['approx_lwb']
    
        self._set_backgrounds()
    
    def _set_backgrounds(self):
        """
        Loop over populations, make separate RB and RS instances for each.
        """
        
        if self.approx_all_xrb * self.approx_all_lwb:
            return
        
        self.Nrbs = self.pops.Npops
        self.rbs = [UniformBackground(pop) for pop in self.pops.pops]
        
        self.all_discrete_lwb = 1
        self.all_discrete_xrb = 1
        for rb in self.rbs:
            self.all_discrete_lwb *= rb.pf['is_lya_src']
            self.all_discrete_xrb *= rb.pf['is_heat_src_igm']
    
    def update_optical_depth(self):
        pass

    def evolve(self, t, dt):
        """
        Evolve radiation background in time.
        
        .. note:: Assumes we're using the generator, otherwise the time 
            evolution must be controlled manually.
            
        """
        
        pass
        
    def step(self):
        pass
        
        
        

