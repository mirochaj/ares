"""

test_chemistry_hydrogen.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Feb 16 12:50:43 MST 2015

Description: 

"""

import ares
import numpy as np
import matplotlib.pyplot as pl

def test():

    pf = \
    {
     'grid_cells': 64,
     'isothermal': True,
     'stop_time': 1e2,
     'radiative_transfer': False,
     'density_units': 1.0,
     'initial_timestep': 1,
     'max_timestep': 1e2,
     'restricted_timestep': None,
     'initial_temperature': np.logspace(3, 5, 64),
     'initial_ionization': [1.-1e-8, 1e-8],        # neutral
    }
    
    sim = ares.simulations.GasParcel(**pf)
    sim.run()
    
    data = sim.history
    
    # Plot last time snapshot
    pl.loglog(data['Tk'][0], data['h_1'][-1,:], color='k')
    pl.loglog(data['Tk'][0], data['h_2'][-1,:], color='k', ls='--')
    
    pl.savefig('%s.png' % (__file__.rstrip('.py')))
    pl.close()    
    
    assert True
    
if __name__ == '__main__':
    test()