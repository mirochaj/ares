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
from ..util.ReadData import _sort_history, _flatten_flux

class MetaGalacticBackground:
    def __init__(self, **kwargs):
        """
        Initialize a MetaGalacticBackground object.    
        """
        
        self.pf = ParameterFile(**kwargs)
        self.field = UniformBackground(**self.pf)

    def update_tau(self):
        pass

    def update_fluxes(self):
        """
        Loop over flux generators.
        """
        
        fluxes = {}
        for i, generator in enumerate(self.field.generators):
            fluxes[self.field.bands[i]] = generator.next()
                
        return fluxes    
            
    def run(self):
        """
        Evolve radiation background in time.

        .. note:: Assumes we're using the generator, otherwise the time 
            evolution must be controlled manually.

        """

        all_fluxes = []
        for fluxes in self.step():
            all_fluxes.append(fluxes)
        
        self.all_fluxes = all_fluxes
        self.history = _sort_history(all_fluxes)
        
    def step(self):
        """
        Initialize generator for the meta-galactic radiation background.
        
        Returns
        -------
        Generator for the background radiation field. Yields the flux for 
        each population.
        
        """
        
        t = 0.0
        z = self.pf['initial_redshift']
        zf = self.pf['final_redshift']
                
        while z > zf:
            
            #tau = self.field.update_optical_depth()
            
            fluxes = self.update_fluxes()
            
                        
                
            yield fluxes    
        
        

