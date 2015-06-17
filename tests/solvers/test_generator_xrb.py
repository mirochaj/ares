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

zi, zf = (40., 10.)

# Initialize radiation background
src_pars = \
{
 # Source properties
 'pop_type': 'galaxy',
 'pop_sfrd': lambda z: 0.01 / (1. + z)**3.,
 'pop_sed': 'pl',
 'pop_alpha': -1.5,
 'pop_Emin': 200.,
 'pop_Emax': 3e4,
 'pop_EminNorm': 5e2,
 'pop_EmaxNorm': 8e3,
 'pop_yield': 2.6e39,
 'pop_yield_units': 'erg/s/SFR',
 
 # Solution method
 'pop_solve_rte': True,
 'pop_tau_Nz': 400,
 
 'sawtooth_nmax': 8,
 
 'initial_redshift': zi,
 'final_redshift': zf,
}

rad1 = ares.simulations.MetaGalacticBackground(**src_pars)
#rad2 = ares.simulations.MetaGalacticBackground(tau_xrb=True, **src_pars)

"""
First, look at background flux itself.
"""

# Compute background flux w/ generator
rad1.run()
#rad2.run()

z1, E1, flux1 = rad1.get_history()

# Compute background flux using more sophisticated (and expensive!) integrator
#Enum = np.logspace(np.log10(2e2), 4.5, 100)
#flux_num = np.array(map(lambda EE: rad.AngleAveragedFlux(zf, EE, zf=zi, 
#    h_2=lambda z: 0.0), Enum))

fig1 = pl.figure(1); ax1 = fig1.add_subplot(111)    
#ax1.semilogx(Enum, flux_num, color='k')
ax1.loglog(E1[0], flux1[-1], color='k', ls='-')
#ax1.loglog(E2, flux2[-1], color='k', ls='--')
ax1.set_xlabel(ares.util.labels['E'])
ax1.set_ylabel(ares.util.labels['flux_E'])

