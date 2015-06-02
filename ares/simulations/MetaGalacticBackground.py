"""

MetaGalacticBackground.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Feb 16 12:43:06 MST 2015

Description: 

"""

import numpy as np
from ..util import ParameterFile
from scipy.interpolate import interp1d
from ..solvers import UniformBackground
from ..util.ReadData import _sort_history, flatten_flux

class MetaGalacticBackground(UniformBackground):
    def __init__(self, grid=None, **kwargs):
        """
        Initialize a MetaGalacticBackground object.    
        """

        self._is_thru_run = False
        
        UniformBackground.__init__(self, grid=grid, **kwargs)

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
        
        self._is_thru_run = True

        all_fluxes_c = []
        all_fluxes_l = []
        for (z, fluxes_c, fluxes_l) in self.step():
            self.update_redshift(z)
            all_fluxes_c.append(fluxes_c)
            all_fluxes_l.append(fluxes_l)

        self.all_fluxes_c = all_fluxes_c
        self.all_fluxes_l = all_fluxes_l
        self.history_c = _sort_history(all_fluxes_c)
        self.history_l = _sort_history(all_fluxes_l)

    def _init_stepping(self):
        """
        Initialize lists which bracket radiation background fluxes.
        
        This is a little tricky for the LWB.
        """
        
        # For "smart" time-stepping
        self._zhi = []; self._zlo = []
        self._fhi = []; self._flo = []
        for i, generator in enumerate(self.generators):
            if generator is None:
                self._fhi.append(None)
                self._flo.append(None)
                continue
                
            z, cflux, lflux = generator.next()    
                
            self._zhi.append(z)    
            self._fhi.append(cflux)
            
            z, cflux, lflux = generator.next()
            
            self._zlo.append(z)
            self._flo.append(cflux)

        self.update_redshift(self._zlo[0])

    def step(self):
        """
        Initialize generator for the meta-galactic radiation background.
        
        ..note:: This can run asynchronously with a MultiPhaseMedium object.

        Returns
        -------
        Generator for the background radiation field. Yields the flux for 
        each population.

        """

        t = 0.0
        z = self.pf['initial_redshift']
        zf = self.pf['final_redshift']
        
        # Start the generator
        while z > zf:     
            z, fluxes_c, fluxes_l = self.update_fluxes()
            
            yield z, fluxes_c, fluxes_l
            
    def update_redshift(self, z):
        self.z = z        
        
    def update_fluxes(self):
        """
        Loop over flux generators and retrieve the next values.
        
        ..note:: Populations need not have identical redshift sampling.
        
        Returns
        -------
        Current redshift and array of fluxes.
        
        """
        
        if (not self._is_thru_run) and (not self.approx_all_sources) and \
            not hasattr(self, '_fhi'):
            self._init_stepping()
        
        fluxes_c = {}  # continuum
        fluxes_l = {}  # line
        for i, generator in enumerate(self.generators):
            
            # Skip approximate (or non-contributing) backgrounds
            if generator is None:
                fluxes_c[i] = None
                fluxes_l[i] = None
                continue
                
            # If not being run as part of another simulation, there are no 
            # external time-stepping constraints, so just poke the generator
            # and move on
            if self._is_thru_run: 
                z, fluxes_c[i], fluxes_l[i] = generator.next()
                continue   
           
            # Otherwise, we potentially need to sub-cycle the background
            
            # For redshifts before this background turns on...
            if self.z > self._zhi[i]:
                z, fluxes_c[i], fluxes_l[i] = \
                    self.z, np.zeros_like(self.energies[i]), 0.0
                continue

            # If we've surpassed the lower redshift bound, poke the 
            # generator
            elif self.z <= self._zlo[i]: 
                
                self._zhi[i] = self._zlo[i]
                self._fhi[i] = self._flo[i]
                z, fluxes_c[i], fluxes_l[i] = generator.next()
                
                # Sometimes the generator's redshift sampling will be finer
                # than needed by e.g., a MultiPhaseMedium, so we cycle
                # multiple times before exiting.
                while z > self.z:
                    self._zhi[i] = self._zlo[i]
                    self._fhi[i] = self._flo[i]

                    z, fluxes_c[i], fluxes_l[i] = generator.next()

                self._zlo[i] = z
                self._flo[i] = fluxes_c[i]
            else:
                z = self.z    

            # If we're between redshift steps, interpolate to find the 
            # background flux
            if self.z != self._zlo[i]:

                z = self.z

                # Interpolate to find flux
                interp = interp1d([self._zlo[i], self._zhi[i]], 
                    [self._flo[i], self._fhi[i]], 
                     axis=0, assume_sorted=True, kind='linear')        

                fluxes_c[i] = interp(z)

        return z, fluxes_c, fluxes_l 

    def update_rate_coefficients(self, z, **kwargs):
        """
        Compute ionization and heating rate coefficients.
        
        Parameters
        ----------
        z : float
            Current redshift.
            
        Returns
        -------
        Dictionary of rate coefficients.
        
        """

        # Must compute rate coefficients from fluxes     
        if self.approx_all_sources:
            kwargs['fluxes'] = [None] * self.Ns
        else:
            z, fluxes_c, fluxes_i = self.update_fluxes()
            kwargs['fluxes'] = fluxes_c    
                    
        # Run update_rate_coefficients within MultiPhaseMedium
        return super(MetaGalacticBackground, self).update_rate_coefficients(z, 
            **kwargs)
                                                                           
    def get_history(self, continuum=True, popid=0):
        """
        Grab data associated with a single population.

        Parameters
        ----------
        continuum : bool
            Get history of continuum emission? If False, retrieves line
            emission history instead.
        popid : int
            ID number for population of interest.

        Returns
        -------
        Tuple containing the redshifts, energies, and fluxes for the given
        population, in that order.
        
        """
        
        if continuum:
            hist = self.history_c
        else:
            hist = self.history_l
        
        return self.redshifts[popid][-1::-1], self.energies[popid], \
            hist[popid]

        