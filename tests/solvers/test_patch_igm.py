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
 #'pop_sfrd': 'robertson2015',
 'pop_sed': 'pl',
 'pop_alpha': -1.5,
 'pop_Emin': 2e2,
 'pop_Emax': 3e4,
 'pop_EminNorm': 5e2,
 'pop_EmaxNorm': 8e3,
 'pop_yield': 2.6e39,
 'pop_yield_units': 'erg/s/sfr',
 'pop_heat_src_igm': True,
 'pop_ion_src_igm': True,
 
 'initial_redshift': 40.,
 'final_redshift': 4.,
 'include_cgm': False,
}

sim = ares.simulations.MultiPhaseMedium(**pars)
sim.run()

mp = ares.analysis.MultiPanel(dims=(2,1))

mp.grid[0].semilogy(sim.history['z'], sim.history['igm_Tk'])
mp.grid[1].semilogy(sim.history['z'], sim.history['igm_h_2'])

for i in range(2):
    mp.grid[i].set_xlim(4.5, 50)

mp.grid[1].set_ylim(1e-10, 1.)
mp.fix_ticks()
pl.draw()
