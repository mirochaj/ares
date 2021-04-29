"""

RestrictTimestep.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Dec 27 14:12:50 2012

Description: 

"""

import numpy as np
from ..util.Warnings import dt_error

tiny_frac = 1e-8
eq_dqdt = 1e-10
huge_dt = 1e30  # seconds

class RestrictTimestep(object):
    def __init__(self, grid, epsilon=0.1, verbose=True):
        self.grid = grid
        self.epsilon = epsilon
        self.verbose = verbose
        self.grid_indices = np.arange(self.grid.dims)
        
    def Limit(self, q, dqdt, z=None, tau=None, tau_ifront=0.5, method=['ions']):
        """
        Limit timestep based on maximum allowed change in fields.  Which 
        fields determined by method parameter.
        """
                
        if method is None:
            return huge_dt        
                
        # Projected timestep for each cell and field (dt.shape = grid x species)
        dt = self.epsilon * q / np.abs(dqdt)
                                 
        # Don't let dt -> 0 where species fraction is zero or one
        dt[np.logical_and(q == 0.0, self.grid.types >= 0)] = huge_dt

        # Don't let dt -> 0 when quantities are in/near equilibrium
        dt[dqdt == 0] = huge_dt
        dt[np.isnan(dqdt)] = huge_dt

        # Isolate cells beyond I-front
        if tau is not None:
            dt[tau <= tau_ifront, ...] = huge_dt

        new_dt = huge_dt
        min_dt = huge_dt
        for mth in method:

            # Determine index correspond to element(s) of q to use to limit dt
            if mth == 'ions':
                j = np.argwhere(self.grid.types == 1).squeeze()
            elif mth == 'neutrals':
                j = np.argwhere(self.grid.types == 0).squeeze()
            elif mth == 'electrons':
                j = self.grid.evolving_fields.index('e')
            elif mth == 'temperature':
                if 'Tk' in self.grid.evolving_fields:
                    j = self.grid.evolving_fields.index('Tk')
                else:
                    continue
            elif mth == 'hubble' and self.grid.expansion:
                min_dt = self.epsilon * self.grid.cosm.HubbleTime(z)
            else:
                raise ValueError('Unrecognized dt restriction method: {!s}'.format(mth))

            if mth != 'hubble':
                min_dt = np.min(dt[..., j])

            not_ok = ((min_dt <= 0) or np.isnan(min_dt) or np.isinf(min_dt))            

            # Determine which cell is behaving badly (if any)
            if not_ok:
                if self.grid.dims == 1:
                    which_cell = 0
                else:
                    if j is not None:
                        cond = np.argwhere(dt[...,j] == min_dt)
                        which_cell = self.grid_indices[cond].squeeze()
                        which_cell = int(which_cell)
                    else:
                        which_cell = 0

                if self.verbose:    
                    dt_error(self.grid, z, q, dqdt, min_dt, which_cell, mth)

            # Update the time-step
            new_dt = min(new_dt, min_dt)

        return new_dt

        
    
     
