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
from ..util.ReadData import _sort_data
from ..util import RestrictTimestep, CheckPoints, ProgressBar, ParameterFile

class GasParcel:
    def __init__(self, **kwargs):
        """
        Initialize a GasParcel object.
        """
        self.pf = ParameterFile(**kwargs)
        
        self.grid = \
        Grid(
            dims=self.pf['grid_cells'], 
            length_units=self.pf['length_units'], 
            start_radius=self.pf['start_radius'],
            approx_Salpha=self.pf['approx_Salpha'], 
            logarithmic_grid=self.pf['logarithmic_grid'],
            )
            
        # Set all properties in one go
        self.grid.set_properties(**self.pf)
        
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
        self.chem = Chemistry(self.grid, rt=self.pf['radiative_transfer'])
        
    def set_rate_coefficients(self, data):
        """
        Compute rate coefficients for next time-step based on current data.
        
        .. note:: This only sets temperature-dependent rate coefficients.
        
        Parameters
        ----------
        data : dict
            Dictionary containing gas properties of each cell.
            
        Returns
        -------
        Nothing -- sets `rate_coefficients` attribute.
            
        """

        rcs = {}

        # First, get coefficients that only depend on kinetic temperature
        if self.grid.isothermal:
            rcs.update(self.chem.rcs)
        else:
            C = self.chem.chemnet.SourceIndependentCoefficients(data['Tk'])
            rcs.update(C)

        self.rate_coefficients = rcs
        
        # Update - add radiative quantities
        self.set_radiation_field()
        
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
            {'Gamma': self.grid.zeros_grid_x_absorbers, 
            'gamma': self.grid.zeros_grid_x_absorbers2,
            'Heat': self.grid.zeros_grid_x_absorbers}
        )

    def evolve(self):
        """
        Run simulation from start to finish.
        """

        pb = ProgressBar(self.pf['stop_time'] * self.pf['time_units'], 
            use=self.pf['progress_bar'])
        pb.start()
        
        # Rate coefficients for initial conditions
        self.set_rate_coefficients(self.grid.data)

        all_data = []
        for t, dt, data in self.step():

            # Re-compute rate coefficients
            self.set_rate_coefficients(data)

            pb.update(t)

            # Save data
            all_data.append(data.copy())

        pb.finish()

        return _sort_data(all_data)        

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
                
        # Evolve in time!
        while t < tf:
                        
            # Evolve by dt
            data = self.chem.Evolve(data, t=t, dt=dt, 
                **self.rate_coefficients)
            
            t += dt 

            # Figure out next dt based on max allowed change in evolving fields
            new_dt = self.timestep.Limit(self.chem.q_grid, 
                self.chem.dqdt_grid, method=self.pf['restricted_timestep'])

            # Limit timestep further based on next DD and max allowed increase
            dt = min(new_dt, 2 * dt)
            dt = min(dt, self.checkpoints.next_dt(t, dt))
            dt = min(dt, max_timestep)
            
            yield t, dt, data

            if t >= tf:
                break
            
    

        
        
