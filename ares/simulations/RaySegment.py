"""

PointSource.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Sep 24 14:55:02 MDT 2014

Description: 

"""

import numpy as np
from ..util import ProgressBar
from .GasParcel import GasParcel
from ..solvers import RadiationField
from ..sources import CompositeSource
from ..util.ReadData import _sort_data

class RaySegment:
    """
    Propagate radiation along a ray!
    """
    def __init__(self, **kwargs):
        """
        Initialize a RaySegment object.
        """
                
        self.parcel = GasParcel(**kwargs)
        
        self.pf = self.parcel.pf
        self.grid = self.parcel.grid
        
        self._set_sources()
        
        # Initialize generator for gas parcel
        self.gen = self.parcel.step()
        
    def _set_sources(self):            
        """
        Initialize radiation source and radiative transfer solver.
        """
    
        if self.pf['radiative_transfer']:
            self.rs = CompositeSource(self.grid, **self.pf)
            allsrcs = self.rs.all_sources
        else:
            allsrcs = None
    
        self.rt = RadiationField(self.grid, allsrcs, **self.pf)        

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
        self.parcel.set_rate_coefficients(self.grid.data)
        self.parcel.set_radiation_field()

        all_t = []
        all_data = []
        for t, dt, data in self.gen:
            
            # Re-compute rate coefficients
            self.parcel.set_rate_coefficients(data)
            
            # Compute ionization / heating rate coefficient
            kw = self.rt.Evolve(data, t, dt)
                        
            # Update rate coefficients accordingly
            self.parcel.rate_coefficients.update(kw)
            
            # Save data
            all_t.append(t)
            all_data.append(data.copy())
            
            if t >= tf:
                break

            pb.update(t)

        pb.finish()

        to_return = _sort_data(all_data)
        to_return['t'] = np.array(all_t)
        
        self.history = to_return

        return to_return

    def step(self, t, dt, data):
        """
        Evolve properties of gas parcel in time.
        """
        
        return self.gen.next()


