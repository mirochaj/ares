"""

test_galaxy.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Tue May 26 14:32:20 MDT 2015

Description: 

"""

import ares
import numpy as np
import matplotlib.pyplot as pl
from ares.physics.Constants import cm_per_mpc, erg_per_ev, g_per_msun, s_per_yr

z = np.arange(10, 35)

# Initialize a GalaxyPopulation
pop = ares.populations.GalaxyPopulation(pop_sed='pl', pop_Emin=2e2, 
    pop_Emax=8e3, pop_EminNorm=5e2, pop_EmaxNorm=8e3, pop_fX=1.0,
    pop_yield=2.6e39, pop_yield_units='erg/s/SFR')
    
# Compute the luminosity density in two bands
Lx1 = np.array(map(pop.LuminosityDensity, z)) * cm_per_mpc**3
Lx2 = np.array(map(lambda z: pop.LuminosityDensity(z, 0.2, 2.), z)) * cm_per_mpc**3

# Plot 'em
pl.semilogy(z, Lx1)
pl.semilogy(z, Lx2)

# Try again with different units
erg_per_phot = pop.src.AveragePhotonEnergy(500., 8e3) * erg_per_ev
y = 2.6e39 * s_per_yr / erg_per_phot 
pop = ares.populations.GalaxyPopulation(pop_sed='pl', pop_Emin=2e2, 
    pop_Emax=8e3, pop_EminNorm=5e2, pop_EmaxNorm=8e3, pop_fX=1.0,
    pop_yield=y, pop_yield_units='photons/Msun')

# Compute the luminosity density in two bands
Lx1 = np.array(map(pop.LuminosityDensity, z)) * cm_per_mpc**3
Lx2 = np.array(map(lambda z: pop.LuminosityDensity(z, 0.2, 2.), z)) * cm_per_mpc**3

# Plot 'em
pl.scatter(z, Lx1)
pl.scatter(z, Lx2)
