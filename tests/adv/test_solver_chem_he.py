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
     'include_He': True,
     'isothermal': True,
     'stop_time': 1e2,
     'radiative_transfer': False,
     'density_units': 1.0,
     'initial_timestep': 1,
     'max_timestep': 1e2,
     'initial_temperature': np.logspace(4, 6, 64),
     'initial_ionization': [1.-1e-8, 1e-8, 1-2e-8, 1e-8, 1e-8], # neutral
    }
    
    sim = ares.simulations.GasParcel(**pf)
    sim.run()
    
    data = sim.history
    
    # Plot last time snapshot
    pl.loglog(data['Tk'][0], data['h_1'][-1,:], color='k')
    pl.loglog(data['Tk'][0], data['h_2'][-1,:], color='k', ls='--')
    
    pl.loglog(data['Tk'][0], data['he_1'][-1,:], color='b')
    pl.loglog(data['Tk'][0], data['he_2'][-1,:], color='b', ls='--')
    pl.loglog(data['Tk'][0], data['he_3'][-1,:], color='b', ls=':')
    pl.ylim(1e-8, 1)

    pl.savefig('%s.png' % (__file__.rstrip('.py')))
    pl.close()

if __name__ == '__main__':
    test()