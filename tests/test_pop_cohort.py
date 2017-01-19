"""

test_pop_cohort.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Thu Jan 19 16:16:28 GMT 2017

Description: Make sure that a pure PopII population (i.e., sources only
at Tmin >= 1e4) can be recovered by a PopIII population (Tmin >= 300 K) but
with a step function in all radiation efficiencies at Tmin=1e4 to eliminate
emission from halos below the step.

"""

import ares
import numpy as np
import matplotlib.pyplot as pl

# First population: sources only in atomic halos
m16 = ares.util.ParameterBundle('mirocha2016:dpl')
pars = m16.pars_by_pop(0, strip_id=True)
pars['pop_Tmin'] = 1.1e4
pop = ares.populations.GalaxyPopulation(**pars)

sim = ares.simulations.Global21cm(**m16)
pop_fs = sim.pops[0] # 'fs' = 'from sim'

# Second population: sources in all halos, but shut off below atomic threshold
m17 = ares.util.ParameterBundle('mirocha2016:dpl')
m17.update(ares.util.ParameterBundle('mirocha2017:step'))
pars2 = m17.pars_by_pop(0, strip_id=True)
pars2['pop_Tmin'] = 1.1e4
pars2['pq_func_par0[1]'] = 0
pars2['pq_func_par0[2]'] = 0
pars2['feedback_LW'] = None
pop2 = ares.populations.GalaxyPopulation(**pars2)  

sim2 = ares.simulations.Global21cm(**m16)
pop2_fs = sim2.pops[0] # 'fs' = 'from sim'  
        
#
## Reference values to use in calculations
z_ref = 20.
Emin_ref = 10.2
Emax_ref = 13.6
tol = 1e-4
##
#

# Test a few quantities above and below the threshold
assert abs(pop.SFRD(z_ref) - pop2.SFRD(z_ref)) <= tol, \
    "Error in SFRD!"

assert abs(pop.Emissivity(z=z_ref, Emin=Emin_ref, Emax=Emax_ref) - \
      pop2.Emissivity(z=z_ref, Emin=Emin_ref, Emax=Emax_ref)) <= tol, \
      "Error in Emissivity!"

# Make sure that the populations extracted from simulation instance are 
# identical to those created inependently.

assert abs(pop.SFRD(z_ref) - pop_fs.SFRD(z_ref)) <= tol, \
    "Error in SFRD!"

assert abs(pop.Emissivity(z=z_ref, Emin=Emin_ref, Emax=Emax_ref) - \
      pop_fs.Emissivity(z=z_ref, Emin=Emin_ref, Emax=Emax_ref)) <= tol, \
      "Error in Emissivity!"

assert abs(pop2.SFRD(z_ref) - pop2_fs.SFRD(z_ref)) <= tol, \
    "Error in SFRD!"

assert abs(pop2.Emissivity(z=z_ref, Emin=Emin_ref, Emax=Emax_ref) - \
      pop2_fs.Emissivity(z=z_ref, Emin=Emin_ref, Emax=Emax_ref)) <= tol, \
      "Error in Emissivity!"


