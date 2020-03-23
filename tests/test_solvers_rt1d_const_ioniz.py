"""

test_const_ionization.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Feb 18 10:30:47 MST 2015

Description: 

"""

import ares
import numpy as np
import matplotlib.pyplot as pl

pf = \
{
 'grid_cells': 64,
 'isothermal': True,
 'stop_time': 1e2,
 'secondary_ionization': False,
 'density_units': 1.0,
 'initial_timestep': 1,
 'max_timestep': 1e2,
 'initial_temperature': np.logspace(3, 5, 64),
 'initial_ionization': [1. - 1e-4, 1e-4],         # neutral
}

def test():

    sim = ares.simulations.GasParcel(radiative_transfer=True, **pf)
    
    # Set constant ionization rate (coefficient) by hand
    k_ion = np.ones([sim.grid.dims,1]) * 1e-14
    
    # Initialize all rate coefficients
    sim.update_rate_coefficients(sim.grid.data)
    sim.set_radiation_field()
    
    # Evolve in time
    all_data = []
    for t, dt, data in sim.step():
    
        sim.update_rate_coefficients(sim.grid.data, k_ion=k_ion)
    
        all_data.append(data)
    
    # Plot ionization state vs. temperature
    pl.loglog(data['Tk'], data['h_1'], color='k', 
        label=r'$\kappa_{\mathrm{ion}} = 10^{-14} \ \mathrm{s}^{-1}$')
    pl.loglog(data['Tk'], data['h_2'], color='k', ls='--')
    
    # Re-run without radiative transfer for comparison
    sim2 = ares.simulations.GasParcel(radiative_transfer=False, **pf)
    sim2.run()
    
    pl.loglog(sim2.history['Tk'][-1], sim2.history['h_1'][-1], color='b', 
        label=r'$\kappa_{\mathrm{ion}} = 0$')
    pl.loglog(sim2.history['Tk'][-1], sim2.history['h_2'][-1], color='b', ls='--')
    
    pl.legend(fontsize=14, loc='best')

    pl.savefig('{!s}.png'.format(__file__[0:__file__.rfind('.')]))
    pl.close()

    assert True

if __name__ == "__main__":
    test()    
    
    
