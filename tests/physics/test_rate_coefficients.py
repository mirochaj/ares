"""

test_physics_rates.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sun Apr 13 16:38:44 MDT 2014

Description: 

"""

import ares, sys
import numpy as np
import matplotlib.pyplot as pl

try:
    species = int(sys.argv[1])
except IndexError:
    species = 0

try:
    species = int(sys.argv[1])
except IndexError:
    species = 0

dims = 32
T = np.logspace(3.5, 6, 500)

fig1 = pl.figure(1); ax1 = fig1.add_subplot(111)
fig2 = pl.figure(2); ax2 = fig2.add_subplot(111)

colors = list('kb')
for i, src in enumerate(['fk94']):

    # Initialize grid object
    grid = ares.static.Grid(dims=dims)
    
    # Set initial conditions
    grid.set_physics(isothermal=True)
    grid.set_chemistry()
    grid.set_density(1)
    grid.set_ionization(Z=1, x=1e-1)
    grid.set_temperature(T)

    coeff = ares.physics.RateCoefficients(grid=grid, rate_src=src, T=T)
    
    # First: collisional ionization, recombination
    CI = map(lambda TT: coeff.CollisionalIonizationRate(species, TT), T)
    RR = map(lambda TT: coeff.RadiativeRecombinationRate(species, TT), T)    
    
    if i == 0:
        labels = [r'$\beta$', r'$\alpha$']
    else:
        labels = [None] * 2
    
    ax1.loglog(T, CI, color=colors[i], ls='-', label=labels[0])
    ax1.loglog(T, RR, color=colors[i], ls='--', label=labels[1])
    
    # Second: Cooling processes
    CIC = map(lambda TT: coeff.CollisionalIonizationCoolingRate(species, TT), T)
    CEC = map(lambda TT: coeff.CollisionalExcitationCoolingRate(species, TT), T)
    RRC = map(lambda TT: coeff.RecombinationCoolingRate(species, TT), T)

    if i == 0:
        labels = [r'$\zeta$', r'$\psi$', r'$\eta$']
    else:
        labels = [None] * 3

    ax2.loglog(T, CIC, color=colors[i], ls='-', label=labels[0])
    ax2.loglog(T, CEC, color=colors[i], ls='--', label=labels[1])
    ax2.loglog(T, RRC, color=colors[i], ls=':', label=labels[2])

ax1.set_ylim(1e-18, 1e-7)
ax1.legend(loc='upper left')  
ax2.legend(loc='lower right')  
ax1.set_xlabel(r'Temperature $(\mathrm{K})$')
ax1.set_ylabel(r'Rate $(\mathrm{cm}^{3} \ \mathrm{s}^{-1})$')
ax2.set_xlabel(r'Temperature $(\mathrm{K})$')
ax2.set_ylabel(r'Rate $(\mathrm{erg} \ \mathrm{cm}^{3} \ \mathrm{s}^{-1})$')
pl.show()
        