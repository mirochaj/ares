"""

run_example_crb_lw.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Jun 29 17:17:37 MDT 2015

Description: 

"""

import ares
import numpy as np
import matplotlib.pyplot as pl
from ares.physics.Constants import erg_per_ev, c, ev_per_hz

pars = \
{
 'pop_type': 'galaxy',
 'pop_sfrd': lambda z: 0.1,
 'pop_sed': 'pl',
 'pop_alpha': 0.0, 
 'pop_Emin': 1.,
 'pop_Emax': 41.8,
 'pop_EminNorm': 13.6,
 'pop_EmaxNorm': 1e2,
 'pop_yield': 1e57,
 'pop_yield_units': 'photons/msun',

 # Solution method
 'pop_solve_rte': True,
 'pop_tau_Nz': 400,
 'include_H_Lya': False,

 'initial_redshift': 40.,
 'final_redshift': 10.,
}

# First calculation: no sawtooth
mgb = ares.simulations.MetaGalacticBackground(**pars)
mgb.run()

z, E, flux = mgb.get_history(flatten=True)
pl.semilogy(E, flux[-1] * E * erg_per_ev, color='k', ls=':')

# Now, turn on sawtooth
pars2 = pars.copy()
pars2['pop_sawtooth'] = True

mgb2 = ares.simulations.MetaGalacticBackground(**pars2)
mgb2.run()

z2, E2, flux2 = mgb2.get_history(flatten=True)
pl.semilogy(E2, flux2[-1] * E2 * erg_per_ev, color='k', ls='--')

# Grab GalaxyPopulation
pop = mgb.pops[0]

# Cosmologically-limited solution to the RTE
# [Equation A1 in Mirocha (2014)]
zi, zf = 40., 10.
e_nu = np.array([pop.Emissivity(zf, EE) for EE in E])
e_nu *= (1. + zf)**4.5 / 4. / np.pi / pop.cosm.HubbleParameter(zf) / -1.5
e_nu *= ((1. + zi)**-1.5 - (1. + zf)**-1.5)
e_nu *= c * ev_per_hz

# Plot it
pl.semilogy(E, e_nu, color='k', ls='-')
pl.xlabel(ares.util.labels['E'])
pl.ylabel(ares.util.labels['flux_E'])
pl.savefig('example_crb_lw.png')
