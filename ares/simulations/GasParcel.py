"""

SingleZone.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Feb 16 12:42:54 MST 2015

Description: This is basically a wrapper for the ChemicalNetwork and Grid, 
which handles grid initialization, time-stepping, data storage, etc.

"""

import numpy as np
from ..static import Grid
from ..solvers import Chemistry
from ..util.ReadData import _sort_history
from ..util import RestrictTimestep, CheckPoints, ProgressBar, ParameterFile

class GasParcel(object):
    def __init__(self, **kwargs):
        """
        Initialize a GasParcel object.
        """

        # This typically isn't the entire parameter file, Grid knows only
        # about a few things.
        self.pf = ParameterFile(**kwargs)

        self.grid = Grid(**self.pf)
        #self.grid = \
        #Grid(
        #    grid_cells=self.pf['grid_cells'], 
        #    length_units=self.pf['length_units'], 
        #    start_radius=self.pf['start_radius'],
        #    approx_Salpha=self.pf['approx_Salpha'],
        #    logarithmic_grid=self.pf['logarithmic_grid'],
        #    cosmological_ics=self.pf['cosmological_ics'],
        #    exotic_heating=self.pf['exotic_heating'],
        #    exotic_heating_func=self.pf['exotic_heating_func'],
        #    )

        # Set all properties in one go
        self.grid.set_properties(**self.pf)
        self.grid.create_slab(**self.pf)

        # Set (optional) additional stuff like radiation field, chemistry, etc.
        self._set_chemistry()

        # For regulating time/redshift steps
        self.checkpoints = CheckPoints(pf=self.pf, 
            grid=self.grid,
            time_units=self.pf['time_units'],
            initial_redshift=self.pf['initial_redshift'],
            final_redshift=self.pf['final_redshift'],
            dzDataDump=self.pf['dzDataDump'],
            dtDataDump=self.pf['dtDataDump'],
            )
        
        # To compute timestep
        self.timestep = RestrictTimestep(self.grid, self.pf['epsilon_dt'], 
            self.pf['verbose'])
        
    def _set_chemistry(self):
        self.chem = Chemistry(self.grid, rt=self.pf['radiative_transfer'],
            recombination=self.pf['recombination'])
        
    def reset(self):
        del self.gen
        self.gen = self.parcel.step()
        
    @property
    def rate_coefficients(self):
        if not hasattr(self, '_rate_coefficients'):
            self._rate_coefficients = self.chem.rcs.copy()
            self.set_radiation_field()
            
        return self._rate_coefficients
        
    def update_rate_coefficients(self, data, **kwargs):
        """
        Compute rate coefficients for next time-step based on current data.
        
        .. note:: This only sets temperature-dependent rate coefficients.
        
        Parameters
        ----------
        data : dict
            Dictionary containing gas properties of each cell.
        kwargs : optional keyword arguments
            ACCEPTS: k_ion, k_ion2, k_heat
            Must be: shape (`grid_cells`, number of absorbing species)
            
        Returns
        -------
        Nothing -- sets `rate_coefficients` attribute.
            
        """

        # First, get coefficients that only depend on kinetic temperature
        if self.grid.isothermal:
            self.rate_coefficients.update(self.chem.rcs)
        else:
            C = self.chem.chemnet.SourceIndependentCoefficients(data['Tk'])
            self.rate_coefficients.update(C)
                        
        self.rate_coefficients.update(kwargs)

    def set_radiation_field(self):
        """
        Compute rate coefficients for next time-step based on current data.
        
        .. note:: This only sets radiation field quantities.
        
        Parameters
        ----------
        data : dict
            Dictionary containing gas properties of each cell.
            
        Returns
        -------
        Nothing -- updates `rate_coefficients` attribute.
        """
        
        self.rate_coefficients.update(
            {'k_ion': self.grid.zeros_grid_x_absorbers, 
             'k_ion2': self.grid.zeros_grid_x_absorbers2,
             'k_heat': self.grid.zeros_grid_x_absorbers},
        )

    def run(self):
        """
        Run simulation from start to finish.
        
        Returns
        -------
        Nothing: sets `history` attribute.
        
        """

        tf = self.pf['stop_time'] * self.pf['time_units']
        pb = ProgressBar(tf, use=self.pf['progress_bar'])
        pb.start()
        
        # Rate coefficients for initial conditions
        self.update_rate_coefficients(self.grid.data)
        self.set_radiation_field()

        all_t = []
        all_data = []
        for t, dt, data in self.step():

            # Re-compute rate coefficients
            self.update_rate_coefficients(data)

            # Save data
            all_t.append(t)
            all_data.append(data.copy())

            if t >= tf:
                break
            
            pb.update(t)

        pb.finish()

        self.history = _sort_history(all_data)
        self.history['t'] = np.array(all_t)
        
    def step(self, t=0., dt=None, tf=None, data=None):
        """
        Evolve properties of gas parcel in time.
        
        Parameters
        ----------
        t : float
            Time at outset of this step [in time_units]
        dt : float
            Step-size
        tf : float
            Final time, i.e., time to stop the calculation.

        Returns
        -------
        Generator for the evolution of this gas parcel. Each iteration yields
        the current time, current time-step, and a dictionary containing all
        grid quantities at this snapshot.

        """    

        if data is None:
            data = self.grid.data.copy()
        if t == 0:
            dt = self.pf['time_units'] * self.pf['initial_timestep']            
        if tf is None:    
            tf = self.pf['stop_time'] * self.pf['time_units']

        max_timestep = self.pf['time_units'] * self.pf['max_timestep']

        self.dt = dt
        self.data = data
                
        # Evolve in time!
        while t < tf:
                        
            # Evolve by dt
            self.data = self.chem.Evolve(self.data, t=t, dt=self.dt, 
                **self.rate_coefficients)
            
            t += self.dt 

            # Figure out next dt based on max allowed change in evolving fields
            new_dt = self.timestep.Limit(self.chem.q_grid, 
                self.chem.dqdt_grid, method=self.pf['restricted_timestep'])

            # Limit timestep further based on next DD and max allowed increase
            dt = min(new_dt, 2 * self.dt)
            dt = min(dt, self.checkpoints.next_dt(t, dt))
            dt = min(dt, max_timestep)
            
            self.dt = dt
            
            yield t, dt, self.data
        
