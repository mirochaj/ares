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
 'radiative_transfer': True,
 'secondary_ionization': False,
 'density_units': 1.0,
 'initial_timestep': 1,
 'max_timestep': 1e2,
 'initial_temperature': np.logspace(3, 5, 64),
 'initial_ionization': [1. - 1e-8, 1e-8],         # neutral
}

gp1 = ares.simulations.GasParcel(**pf)
gp2 = ares.simulations.GasParcel(**pf)

gp1.set_rate_coefficients(gp1.grid.data)
gp2.set_rate_coefficients(gp2.grid.data)

gen1 = gp1.step()
gen2 = gp2.step()

Gamma = [1e-14, 1e-15]
gps = [gp1, gp2]

all_data = []

t = 0
while t < 1e15:
    all_data = []
    for i, gen in enumerate([gen1, gen2]):

        gp = gps[i]
        t, dt, data = gen.next()

        gp.set_rate_coefficients(gp.grid.data)

        G = np.ones([gp.grid.dims,1]) * Gamma[i]
        gp.rate_coefficients.update({'Gamma': G})

        all_data.append(data)
        
data1, data2 = all_data
        
pl.loglog(data1['Tk'], data1['h_1'], color='k')
pl.loglog(data1['Tk'], data1['h_2'], color='k', ls='--')

pl.loglog(data2['Tk'], data2['h_1'], color='b')
pl.loglog(data2['Tk'], data2['h_2'], color='b', ls='--')
    
    
