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
from ..util.ReadData import _sort_history, flatten_energies

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

        all_z = []         # sometimes not deterministic
        all_fluxes_c = []
        all_fluxes_l = []
        for (z, fluxes_c, fluxes_l) in self.step():
            self.update_redshift(z)
            all_z.append(z)
            all_fluxes_c.append(fluxes_c)
            all_fluxes_l.append(fluxes_l)

        self.all_z = all_z
        self.all_fluxes_c = all_fluxes_c
        self.all_fluxes_l = all_fluxes_l

        self._history_c = _sort_history(all_fluxes_c)
        self._history_l = _sort_history(all_fluxes_l)

    def _init_stepping(self):
        """
        Initialize lists which bracket radiation background fluxes.
        
        The structure of these lists is as follows:
        (1) Each list contains one element per source population.
        (2) If that population will approximate the RTE, this entry will be 
            None.
        (3) The redshift lists, _zlo and _zhi, will just be a sequences of 
            floats. 
        (4) The flux entires, if not None, will be lists, since in general an
            emission band can be broken up into several pieces. In this case,
            the number of entries (for each source population) will be equal
            to the number of bands, which you can find in self.bands_by_pop.
        
        Sets
        ----
        Several attributes:
        (1) _zhi, _zlo
        (2) _fhi, _flo
        
        """
                
        # For "smart" time-stepping
        self._zhi = []; self._zlo = []
        self._fhi = []; self._flo = []
        for i, generator in enumerate(self.generators):
            
            # Recall that each generator may actually be a list of generators,
            # one for each (sub-)band.
            
            if (generator == [None]) or (generator is None):
                self._zhi.append(None)
                self._zlo.append(None)
                self._fhi.append(None)
                self._flo.append(None)
                continue
                
            # Only make it here when real RT is happenin'    
              
            _fhi = []
            _flo = []    
            for j in range(len(generator)):
                _fhi.append(np.zeros_like(self.energies[i][j]))
                _flo.append(np.zeros_like(self.energies[i][j]))
            
            # Loop over sub-bands and retrieve fluxes
            for j, gen in enumerate(generator):
                
                if gen is None:
                    raise ValueError('this should never happen')
                                                        
                # Tap generator, grab fluxes
                zhi, cflux, lflux = gen.next()
                                
                # Increment the flux
                _fhi[j] += cflux.copy()
                
                # Tap generator, grab fluxes (again)
                zlo, cflux, lflux = gen.next()
                                                                    
                # Increment the flux
                _flo[j] += cflux.copy()
                
            # Save fluxes for this population
            self._zhi.append(zhi)
            self._zlo.append(zlo)
            
            self._fhi.append(_fhi)
            self._flo.append(_flo)
            
        # Set the redshift based on whichever population took the smallest
        # step. Other populations will interpolate to find flux.
        self.update_redshift(max(self._zlo))
        
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

    @property
    def history_c(self):
        if hasattr(self, '_history_c'):
            pass
        elif hasattr(self, 'all_fluxes_c'):
            self._history_c = _sort_history(self.all_fluxes_c)
        else:
            raise NotImplemented('help!')    
    
        return self._history_c
    
    @property
    def history_l(self):
        if hasattr(self, '_history_l'):
            pass
        elif hasattr(self, 'all_fluxes_l'):
            self._history_l = _sort_history(self.all_fluxes_l)
        else:
            raise NotImplemented('help!')    
    
        return self._history_l
        
    def update_fluxes(self):
        """
        Loop over flux generators and retrieve the next values.
        
        ..note:: Populations need not have identical redshift sampling.

        Returns
        -------
        Current redshift and dictionary of fluxes. Each element of the flux
        dictionary corresponds to a single population, and within that, there
        are separate lists for each sub-band over which we solve the RTE.
        
        """
                        
        if (not self._is_thru_run) and (not self.approx_all_pops) and \
            not hasattr(self, '_fhi'):
            
            self._init_stepping()
                        
            # Save fluxes by pop as simulations run
            self.all_z = []
            self.all_fluxes_c = []
            self.all_fluxes_l = []
        
        z_by_pop = [None for i in range(self.Npops)]
        fluxes_c = {}  # continuum
        fluxes_l = {}  # line
        for i, pop_generator in enumerate(self.generators):
            
            # Skip approximate (or non-contributing) backgrounds
            if pop_generator is None:
                fluxes_c[i] = None
                fluxes_l[i] = None
                continue
                                    
            fluxes_c_by_band = []
            fluxes_l_by_band = []

            # For each population, the band is broken up into pieces
            for j, generator in enumerate(pop_generator):
                                                            
                # If not being run as part of another simulation, there are 
                # no external time-stepping constraints, so just poke the 
                # generator and move on
                if self._is_thru_run:
                    z, fc, fl = generator.next()
                    fluxes_c_by_band.append(fc)
                    fluxes_l_by_band.append(fl)
                    continue
                    
                # Otherwise, we potentially need to sub-cycle the background.
                # This may happen if (1) the time-step is being regulated
                # from the simulation in which this background is embedded 
                # (i.e., epsilon_dt requires smaller timestep than redshift
                # step allowed by this population) or (2) if other populations
                # have a different requirement for the redshift sampling, 
                # such that this population must interpolate between its
                # (larger) redshift steps while other populations churn away.

                # For redshifts before this background turns on...
                # (this should only happen once)
                if self.z > self._zhi[i]:
                    z, fc, fl = \
                        self.z, np.zeros_like(self.energies[i][j]), 0.0
                    fluxes_c_by_band.append(fc)
                    fluxes_l_by_band.append(fl)
                    continue

                # If we've surpassed the lower redshift bound, poke the 
                # generator
                elif self.z <= self._zlo[i]:

                    self._zhi[i] = self._zlo[i]
                    self._fhi[i][j] = self._flo[i][j]
                    z, fc, fl = generator.next()
                    
                    # Sometimes the generator's redshift sampling will be finer
                    # than needed by e.g., a MultiPhaseMedium, so we cycle
                    # multiple times before exiting.
                    while z > self.z:
                        self._zhi[i] = self._zlo[i]
                        self._fhi[i][j] = self._flo[i][j]
                
                        z, fc, fl = generator.next()
                        
                    self._zlo[i] = z
                    self._flo[i][j] = fc
                else:
                    z = self.z

                # If we're between redshift steps, interpolate to find the 
                # background flux
                if self.z != self._zlo[i]:

                    z = self.z

                    # Interpolate to find flux
                    interp = interp1d([self._zlo[i], self._zhi[i]], 
                        [self._flo[i][j], self._fhi[i][j]], 
                         axis=0, assume_sorted=True, kind='linear')        

                    fc = interp(z)

                fl = 0.0    

                fluxes_c_by_band.append(fc)
                fluxes_l_by_band.append(fl)    

            z_by_pop[i] = z    
            fluxes_c[i] = fluxes_c_by_band
            fluxes_l[i] = fluxes_l_by_band

        if not self._is_thru_run:
            self.all_z.append(z_by_pop)
            self.all_fluxes_c.append(fluxes_c)
            self.all_fluxes_l.append(fluxes_l)

        return max(z_by_pop), fluxes_c, fluxes_l

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
        if self.approx_all_pops:
            kwargs['fluxes'] = [None] * self.Npops
        else:    
            z, fluxes_c, fluxes_i = self.update_fluxes()
            kwargs['fluxes'] = fluxes_c    
        
        # Run update_rate_coefficients within MultiPhaseMedium
        return super(MetaGalacticBackground, self).update_rate_coefficients(z, 
            **kwargs)

    def get_history(self, continuum=True, popid=0, flatten=False):
        """
        Grab data associated with a single population.

        Parameters
        ----------
        continuum : bool
            Get history of continuum emission? If False, retrieves line
            emission history instead.
        popid : int
            ID number for population of interest.
        flatten : bool
            For sawtooth calculations, the energies are broken apart into 
            different bands which have different sizes. Set this to true if
            you just want a single array, rather than having the energies
            and fluxes broken apart by their band.

        Returns
        -------
        Tuple containing the redshifts, energies, and fluxes for the given
        population, in that order.
        
        if flatten == True:
            The energy array is 1-D.
            The flux array will have shape (z, E)
        else:
            The energies are stored as a list. The number of elements will
            be determined by how many sub-bands there are. Each element will
            be a list or an array, depending on whether or not there is a 
            sawtooth component to that particular background.
        
        """
        
        if continuum:
            hist = self.history_c
        else:
            hist = self.history_l
            
        if self._is_thru_run:
            z = self.redshifts[popid]
        else:
            # This may change on the fly due to sub-cycling and such
            z = np.array(self.all_z).T[popid][-1::-1]
        
        if flatten:
            E = flatten_energies(self.energies[popid])
                        
            f = np.zeros([len(z), E.size])
            for i, flux in enumerate(hist[popid]):
                fzflat = []
                for j in range(len(self.energies[popid])):
                    fzflat.extend(flux[j])
                
                f[i] = np.array(fzflat)

            return z[-1::-1], E, np.array(f)
        else:
            return z[-1::-1], self.energies[popid], hist[popid]

        