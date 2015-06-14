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

zi, zf = (7., 2.7)

# Initialize radiation background
src_pars = \
{
 # Source properties
 'pop_type': 'galaxy',
 'pop_sfrd': lambda z: 0.01 / (1. + z)**3., # Use R15 or HM12
 'pop_sed': 'sazonov2004',
 'pop_alpha': 1.0,
 'pop_Emin': 1.0,
 'pop_Emax': 1e4,
 'pop_EminNorm': 5e2,
 'pop_EmaxNorm': 8e3,
 'pop_yield': 2.6e39,
 'pop_yield_units': 'erg/s/SFR',
 
 # Solution method
 'pop_solve_rte': True,
 'pop_tau_Nz': 400,
 'include_H_Lya': False,
 
 'sawtooth_nmax': 8,
 
 'initial_redshift': zi,
 'final_redshift': zf,
}

rad1 = ares.simulations.MetaGalacticBackground(pop_sawtooth=True, 
    approx_tau=None, **src_pars)
rad2 = ares.simulations.MetaGalacticBackground(pop_sawtooth=False, 
    approx_tau=None, **src_pars)
rad3 = ares.simulations.MetaGalacticBackground(pop_sawtooth=True,
    approx_tau=True, **src_pars)
rad4 = ares.simulations.MetaGalacticBackground(pop_sawtooth=True,
    approx_tau='post_EoR', **src_pars)

# Compute background flux
rad1.run()
rad2.run()
rad3.run()
rad4.run()

z1, E1, flux1 = rad1.get_history(flatten=True)
z2, E2, flux2 = rad2.get_history(flatten=True)
z3, E3, flux3 = rad3.get_history(flatten=True)
z4, E4, flux4 = rad4.get_history(flatten=True)

pl.loglog(E1, E1 * flux1[-1] * erg_per_ev, color='k')
pl.loglog(E2, E2 * flux2[-1] * erg_per_ev, color='b', ls='--', lw=3)
pl.loglog(E3, E3 * flux3[-1] * erg_per_ev, color='r', ls=':')
pl.loglog(E4, E4 * flux4[-1] * erg_per_ev, color='g', ls='--')
pl.xlabel(ares.util.labels['E'])
pl.ylabel(ares.util.labels['flux_E'])

