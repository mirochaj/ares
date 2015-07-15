"""

test_generator_xrb.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri Jun 14 09:20:22 2013

Description:

"""

import ares
import numpy as np
import matplotlib.pyplot as pl
from ares.physics.Constants import erg_per_ev, c, ev_per_hz

# Initialize radiation background
pars = \
{
 # Source properties
 'pop_type': 'galaxy',
 'pop_sfrd': lambda z: 0.1,
 'pop_sed': 'pl',
 'pop_alpha': -1.5,
 'pop_Emin': 1e2,
 'pop_Emax': 3e4,
 'pop_EminNorm': 5e2,
 'pop_EmaxNorm': 8e3,
 'pop_yield': 2.6e39,
 'pop_yield_units': 'erg/s/sfr',

 'pop_solve_rte': True,
 "pop_tau_Nz": 400,

 'initial_redshift': 40.,
 'final_redshift': 10.,
}


mgb = ares.simulations.MetaGalacticBackground(**pars)
mgb.run()

"""
First, look at background flux itself.
"""

z, E, flux = mgb.get_history(flatten=True)

pl.loglog(E, flux[-1] * E * erg_per_ev, color='k', ls='--')

# Grab GalaxyPopulation
pop = mgb.pops[0]

# Cosmologically-limited solution to the RTE
# [Equation A1 in Mirocha (2014)]
zi, zf = 40., 10.
e_nu = np.array(map(lambda E: pop.Emissivity(10., E), E))
e_nu *= c / 4. / np.pi / pop.cosm.HubbleParameter(10.) 
e_nu *= (1. + 10.)**6. / -3.
e_nu *= ((1. + 40.)**-3. - (1. + 10.)**-3.)
e_nu *= ev_per_hz

# Plot it
pl.loglog(E, e_nu, color='k', ls='-')
pl.xlabel(ares.util.labels['E'])
pl.ylabel(ares.util.labels['flux_E'])
pl.savefig('example_crb_xr.png')


"""
Do neutral absorption in IGM.
"""

pars['pop_solve_rte'] = True
pars['approx_tau'] = 'neutral'
pars['pop_tau_Nz'] = 400

mgb = ares.simulations.MetaGalacticBackground(**pars)
mgb.run()

z, E, flux = mgb.get_history(flatten=True)

pl.loglog(E, flux[-1] * E * erg_per_ev, color='k', ls=':')


