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
 'pop_sed': 'pl',
 'pop_alpha': 1.0,
 'pop_Emin': 13.6,
 'pop_Emax': 24.6,
 'pop_EminNorm': 13.6,
 'pop_EmaxNorm': 24.6,
 'pop_yield': 1e57,
 'pop_yield_units': 'photons/msun',
 'pop_heat_src_igm': False,
 'pop_ion_src_igm': False,
 'initial_redshift': 30.,
 'final_redshift': 7.,
}

sim = ares.simulations.MultiPhaseMedium(**pars)
sim.run()

pl.semilogy(sim.history['z'], sim.history['cgm_h_2'])

