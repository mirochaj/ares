"""

run_example_crb_uv.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Jul  2 17:20:11 MDT 2015

Description: 

"""

import ares
import numpy as np
import matplotlib.pyplot as pl
from ares.physics.Constants import erg_per_ev

# Initialize radiation background
pars = \
{
 # Source properties
 'pop_type': 'galaxy',
 'pop_sfrd': lambda z: 0.1,
 'pop_sed': 'pl',
 'pop_alpha': 1.0,
 'pop_Emin': 13.6,
 'pop_Emax': 1e2,
 'pop_EminNorm': 13.6,
 'pop_EmaxNorm': 1e2,
 'pop_yield': 1e57,
 'pop_yield_units': 'photons/msun',

 # Solution method
 'pop_solve_rte': True,
 'pop_tau_Nz': 400,
 'include_H_Lya': False,

 'sawtooth_nmax': 8,
 'pop_sawtooth': True,

 'initial_redshift': 7.,
 'final_redshift': 3.,
}


mgb = ares.simulations.MetaGalacticBackground(**pars)
mgb.run()

"""
First, look at background flux itself.
"""

z, E, flux = mgb.get_history(flatten=True)

pl.semilogy(E, flux[-1] * E * erg_per_ev, color='k')
pl.xlabel(ares.util.labels['E'])
pl.ylabel(ares.util.labels['flux_E'])
pl.savefig('example_crb_uv.png')


