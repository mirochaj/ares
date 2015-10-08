"""

test_21cm_cxrb.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Nov 11 13:02:00 MST 2013

Description: 

"""

import os, ares

pars1 = \
{
 'pop_solve_rte{1}': True,
 'pop_tau_Nz{1}': 400,
 'pop_approx_tau{1}': True,
}

pars2 = pars1.copy()
pars2['pop_approx_tau{1}'] = 'neutral'

ax = None

# Examine effects of realistic CXRB (limits of neutral medium and optically thin)
for p in [pars1, pars2, {'problem_type': 101}]:
    sim = ares.simulations.Global21cm(**p)
    sim.run()
    
    anl = ares.analysis.Global21cm(sim)
    ax = anl.GlobalSignature(ax=ax)





