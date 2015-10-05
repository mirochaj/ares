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
 'pop_alpha': 1.0,
 'pop_Emin': 1.,
 'pop_Emax': 500.,
 'pop_EminNorm': 10.2,
 'pop_EmaxNorm': 13.6,
 'pop_yield': 9690,
 'pop_yield_units': 'photons/baryon',
 
 # Solution method
 'pop_solve_rte': True,
 'pop_tau_Nz': 400,
 'include_H_Lya': False,

 'sawtooth_nmax': 8,

 'initial_redshift': zi,
 'final_redshift': zf,
}

rad1 = ares.simulations.MetaGalacticBackground(pop_sawtooth=True, **src_pars)
rad2 = ares.simulations.MetaGalacticBackground(pop_sawtooth=False, **src_pars)

"""
First, look at background flux itself.
"""

# Compute background flux w/ generator
rad1.run()
rad2.run()

z1, E1, flux1 = rad1.get_history(flatten=True)
z2, E2, flux2 = rad2.get_history(flatten=True)

pl.loglog(E1, flux1[-1], color='k')
pl.loglog(E2, flux2[-1], color='b', ls='--')

