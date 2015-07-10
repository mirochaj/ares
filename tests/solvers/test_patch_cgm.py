"""

test_multi_phase.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sat Feb 21 11:55:00 MST 2015

Description: 

"""

import ares
import numpy as np
import matplotlib.pyplot as pl

pars = \
{
 'pop_type': 'galaxy',
 'pop_sfrd': 'robertson2015',
 'pop_Emin': 13.6,
 'pop_Emax': 24.6,
 'pop_EminNorm': 13.6,
 'pop_EmaxNorm': 24.6,
 'pop_yield': 1e54,
 'pop_yield_units': 'photons/s/sfr',
 'pop_heat_src_igm': False,
 'pop_ion_src_igm': False,
 'initial_redshift': 30.,
 'final_redshift': 4.,
 'include_igm': False,
 'cgm_Tk': 1e4,
 'clumping_factor': 3.,
}

sim = ares.simulations.MultiPhaseMedium(**pars)
sim.run()

pl.plot(sim.history['z'], sim.history['cgm_h_1'])
pl.xlim(4.5, 30)
pl.ylim(1e-4, 1)
pl.ylabel(r'$1 - Q_{\mathrm{HII}}$')
pl.xlabel(r'$z$')

