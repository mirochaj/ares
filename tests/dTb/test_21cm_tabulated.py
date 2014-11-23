"""

test_21cm_parameterized.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Aug  6 08:54:15 MDT 2014

Description: 21-cm signal in absence of astrophysical sources.

"""

import ares
import numpy as np
import matplotlib.pyplot as pl
from ares.analysis.MultiPlot import MultiPanel

# Create instance of Hydrogen class
hydr = ares.physics.Hydrogen()

# Load CosmoRec solution
CR = ares.util.ReadData.load_inits()
z, Tk, xe = CR['z'], CR['Tk'], CR['xe']

# Compute hydrogen and electron densities
nH = hydr.cosm.nH(CR['z'])
ne = nH * xe

# Assume xHII = xHeII
xHII = xe / (1. + hydr.cosm.y)

# Spin temperature: middle argument is Lyman-alpha flux (0 here)
Ts = hydr.SpinTemperature(CR['z'], CR['Tk'], 0.0, xHII, ne)

# Brightness temperature
dTb = hydr.DifferentialBrightnessTemperature(z, xHII, Ts)

# Start plotting stuff
mp = MultiPanel(dims=(2, 1), panel_size=(1, 0.5))

# Plot temperatures
mp.grid[1].loglog(z, map(hydr.cosm.TCMB, z), color='k', ls=':')
mp.grid[1].loglog(z, Tk, color='k', ls='--')
mp.grid[1].loglog(z, Ts, color='k', ls='-')

# Plot 21-cm signature
mp.grid[0].semilogx(z, dTb, color='k', label='analytic')

# Labels
mp.grid[0].set_xlabel(r'$z$')
mp.grid[1].set_ylabel('Temperature')
mp.grid[1].set_xticklabels([])
mp.grid[1].set_ylabel(r'$\delta T_b \ (\mathrm{mK})$')

# Limits
mp.grid[0].set_xlim(10, z.max())
mp.grid[1].set_xlim(10, z.max())
mp.grid[1].set_ylim(1, 1e4)
mp.grid[0].set_ylim(-40, 5)

pl.draw()

