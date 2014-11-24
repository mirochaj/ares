"""

test_helium_chemistry.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Dec 26 18:37:48 2012

Description: Evolve helium species.

"""

import ares
import numpy as np
import matplotlib.pyplot as pl

dims = 64
T = np.logspace(3.5, 6, dims)

# Initialize grid object
grid = ares.static.Grid(dims=dims)

# Set initial conditions
grid.set_physics(isothermal=True)
grid.set_chemistry(include_He=True)
grid.set_density(nH=1.)
grid.set_temperature(T)
grid.set_ionization(state='neutral')

# Initialize chemistry network / solver
chem = ares.solvers.Chemistry(grid, rt=False)

# Compute rate coefficients once (isothermal)
chem.chemnet.SourceIndependentCoefficients(grid.data['Tk'])

# To compute timestep
timestep = ares.util.RestrictTimestep(grid)

# Evolve chemistry
data = grid.data
dt = ares.physics.Constants.s_per_myr * 1
dt_max = 1e2 * ares.physics.Constants.s_per_myr
t = 0.0
tf = ares.physics.Constants.s_per_gyr

# Initialize progress bar
pb = ares.util.ProgressBar(tf)
pb.start()

# Start calculation
while t < tf:
    pb.update(t)
    data = chem.Evolve(data, t=t, dt=dt)
    t += dt 
    
    # Limit time-step based on maximum rate of change in grid quantities
    new_dt = timestep.Limit(chem.chemnet.q, chem.chemnet.dqdt)
    
    # Limit to factor of 2x increase in timestep
    dt = min(new_dt, 2 * dt)
    
    # Impose maximum timestep
    dt = min(dt, dt_max)
    
    # Make sure we end at t == tf
    dt = min(dt, tf - t)

pb.finish() 

# Plot solution
ax = pl.subplot(111)

ax.loglog(T, data['h_1'], color='b', ls='-', label='H')
ax.loglog(T, data['h_2'], color='b', ls='--')

ax.loglog(T, data['he_1'], color='k', ls='-', label='He')
ax.loglog(T, data['he_2'], color='k', ls='--')
ax.loglog(T, data['he_3'], color='k', ls=':')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$T \ (\mathrm{K})$')
ax.set_ylabel('Species Fraction')
ax.set_xlim(min(T), max(T))
ax.set_ylim(5e-9, 1.5)
pl.legend(loc='upper right', frameon=False)
pl.draw()


