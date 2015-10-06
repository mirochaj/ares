"""

test_21cm_cxrb.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Nov 11 13:02:00 MST 2013

Description: 

"""

import os, ares

pars = \
{
 'pop_solve_rte{1}': True,
 'redshift_bins{1}': 400,
}

# Multi-pop model, one with real RT
for p in [pars]:
    sim = ares.simulations.Global21cm(**p)
    sim.run()
    
    anl = ares.analysis.Global21cm(sim)
    ax = anl.GlobalSignature()

