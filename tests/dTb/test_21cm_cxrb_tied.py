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
 'Tmin{0}': 1e4,
 'source_type{0}': 'star',
 'source_temperature{0}': 1e4,
 'fstar{0}': 5e-2,
 'Nion{0}': 4e3,
 'Nlw{0}': 9690.,
 'is_heat_src_igm{0}': False,
 'is_ion_src_cgm{0}': True,
 'is_ion_src_igm{0}': False,
 'norm_by{0}': 'lw',
 'approx_lya{0}': True,
 
 'Tmin{1}': 'Tmin{0}',
 'source_type{1}': 'bh',
 'fstar{1}': 'fstar{0}',
 'fX{1}': 1.,
 'norm_by{1}': 'xray',
 'is_lya_src{1}': False,
 'is_ion_src_cgm{1}': False,
 'is_ion_src_igm{1}': True,
 'is_heat_src_igm{1}': True,
 'approx_xray{1}': False,
 'load_tau{1}': True,
 'redshift_bins{1}': 400,
 'spectrum_type{1}': 'pl',
 'spectrum_alpha{1}': -1.5,
 'spectrum_Emin{1}': 2e2,
 'spectrum_Emax{1}': 3e4,
}

# Multi-pop model, one with real RT
sim = ares.simulations.Global21cm(**pars)
sim.run()

anl = ares.analysis.Global21cm(sim)
ax = anl.GlobalSignature()

