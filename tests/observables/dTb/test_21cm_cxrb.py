"""

test_21cm_cxrb.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Nov 11 13:02:00 MST 2013

Description: 

"""

import os, ares

src1 = \
{
 'source_type': 'star',
 'source_temperature': 1e4,
 'fstar': 1e-1,
 'Nion': 4e3,
 'Nlw': 9690.,
 'is_ion_src_cgm': True,
 'is_ion_src_igm': False,
 'norm_by': 'lw',
 'approx_lya': True,
}

sed1 = {}

src2 = \
{
 'source_type': 'bh',
 'fstar': 1e-1,
 'fX': 1.,
 'norm_by': 'xray',
 'is_lya_src': False,
 'is_ion_src_cgm': False,
 'is_ion_src_igm': True,
 'approx_xray': False,
 'load_tau': True,
 'redshift_bins': 400,
}

sed2 = \
{
 'spectrum_type': 'pl',
 'spectrum_alpha': -1.5,
 'spectrum_Emin': 2e2,
 'spectrum_Emax': 3e4,
}

pars = \
{
 'initial_redshift': 50,
 'final_redshift': 6,
 'source_kwargs': [src1, src2],
 'spectrum_kwargs': [sed1, sed2],
}

# Multi-pop model, one with real RT
sim = ares.simulations.Global21cm(**pars)
sim.run()

anl = ares.analysis.Global21cm(sim)
ax = anl.GlobalSignature()

