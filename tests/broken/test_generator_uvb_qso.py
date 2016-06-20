"""

test_generator_xrb.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri Jun 14 09:20:22 2013

Description:

"""

import numpy as np
import os, sys, ares
import matplotlib.pyplot as pl
from ares.physics.Constants import erg_per_ev

zi, zf = (6, 3.)

# Initialize radiation background
src_pars = \
{
 # Source properties
 'pop_type': 'galaxy',
 'pop_sed': 'sazonov2004',
 'pop_rhoL': 'ueda2003',
 'pop_kwargs': {'evolution': 'ple'},
 
 #'pop_alpha': 1.0,
 'pop_Emin': 1.0,
 'pop_Emax': 5e4,
 'pop_EminNorm': 2e3,
 'pop_EmaxNorm': 1e4,

 # Solution method
 'pop_solve_rte': True,
 'pop_tau_Nz': 100,
 'include_H_Lya': False,
 
 'sawtooth_nmax': 8,
 
 'initial_redshift': zi,
 'final_redshift': zf,
}

rad1 = ares.simulations.MetaGalacticBackground(pop_sawtooth=True, 
    approx_tau=True, **src_pars)

# Compute background flux
rad1.run()

# Grab background flux from Haardt & Madau (2012)
hm12 = ares.util.read_lit('haardt2012')
z, E, flux = hm12.MetaGalacticBackground()

# Plot it at z=3
ax = None
colors = 'k', 'b', 'r'
for i, rad in enumerate([rad1]):
    anl = ares.analysis.MetaGalacticBackground(rad)
    ax = anl.PlotBackground(z=3, ax=ax, color=colors[i])
    
    j = np.argmin(np.abs(z - 3.))
    ax.plot(E, flux[j] / 1e-21, color='c')

ax.set_xlim(1, 4e3)
ax.set_ylim(1e-5, 2e2)
ax.set_title(r'$z=3$')
pl.draw()

