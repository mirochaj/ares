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
 'source_sed{0}': 'bb',
 'source_temperature{0}': 1e4,
 'fstar{0}': 1e-1,
 'Nion{0}': 4e3,
 'Nlw{0}': 9690.,
 'is_heat_src_igm{0}': False,
 'is_ion_src_cgm{0}': True,
 'is_ion_src_igm{0}': False,
 'norm_by{0}': 'lw',
 'approx_lwb{0}': True,

 'fstar{1}': 1e-1,
 'fX{1}': 1.,
 'norm_by{1}': 'xray',
 'is_lya_src{1}': False,
 'is_ion_src_cgm{1}': False,
 'is_ion_src_igm{1}': True,
 'is_heat_src_igm{1}': True,
 'approx_xrb{1}': False,
 'discrete_xrb{1}': True,
 'redshift_bins{1}': 400,
 'source_sed{1}': 'pl',
 'source_alpha{1}': -1.5,
 'source_Emin{1}': 2e2,
 'source_Emax{1}': 3e4,
}

# Multi-pop model, one with real RT
sim = ares.simulations.Global21cm(problem_type=101, **pars)
sim.run()

anl = ares.analysis.Global21cm(sim)
ax = anl.GlobalSignature()

