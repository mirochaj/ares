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
from ..util.ReadData import _sort_history, flatten_flux

class MetaGalacticBackground:
    def __init__(self, **kwargs):
        """
        Initialize a MetaGalacticBackground object.    
        """

        self.pf = ParameterFile(**kwargs)
        self.field = UniformBackground(**self.pf)

    def run(self):
        """
        Evolve radiation background in time.

        .. note:: Assumes we're using the generator, otherwise the time 
            evolution must be controlled manually.

        Returns
        -------
        Nothing: sets `history` attribute containing the entire evolution
        of the background for each population.

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
            fluxes = self.update_fluxes()

            yield fluxes

    def update_fluxes(self):
        """
        Loop over flux generators and retrieve the next values.
        """

        fluxes = {}
        for i, generator in enumerate(self.field.generators):
            if generator is None:
                fluxes[i] = None
                continue

            fluxes[i] = generator.next()

        return fluxes    
            
    def get_history(self, popid=0):
        """
        Grab data associated with a single population.
        
        Parameters
        ----------
        popid : int
            ID number for population of interest.
        
        Returns
        -------
        Tuple containing the redshifts, energies, and fluxes for the given
        population.
        
        """
        return self.field.redshifts[popid][-1::-1], self.field.energies[popid], \
            self.history[popid]

        