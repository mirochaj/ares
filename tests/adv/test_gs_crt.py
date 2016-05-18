"""

test_gs_crt.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Wed May 11 09:46:05 PDT 2016

Description: 

"""

import ares
import numpy as np
import matplotlib.pyplot as pl

pars = \
{

 'pop_sed{1}': 'mcd',
 'pop_alpha{1}': -1.5,
 'pop_Emin{1}': 2e2,
 'pop_Emax{1}': 3e4,
 'pop_EminNorm{1}': 5e2,
 'pop_EmaxNorm{1}': 8e3,
 #'pop_logN{1}': -np.inf,

 'pop_solve_rte{1}': True,
 'pop_tau_Nz{1}': 1e3,
 'pop_approx_tau{1}': 'neutral',
 
 # Force optically thin to overestimate heating/ionization?
 
 'final_redshift': 5,
 'initial_redshift': 50,
 'problem_type': 101.2
}

ax1 = None; ax2 = None
labels = ['fidicual', 'fiducial+RT', 'fiducial+OTRT']
for i, solve_rte in enumerate([False, True, True]):
    
    pars['pop_solve_rte{1}'] = solve_rte
    
    if i == 2:
        pars['pop_approx_tau{1}'] = True
    
    sim = ares.simulations.Global21cm(**pars)
    sim.run()
    
    ax1 = sim.GlobalSignature(fig=1, label=labels[i], ax=ax1)
    ax2 = sim.IonizationHistory(fig=2, ax=ax2)
    
ax1.legend(loc='lower right')
pl.show()    
    
for i in range(1,3):
    pl.figure(i)
    pl.savefig('%s_%i.png' % (__file__.rstrip('.py'), i))
#pl.close()
