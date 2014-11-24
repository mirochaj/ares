"""

test_21cm_fiducial.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Sep  8 10:27:15 MDT 2014

Description: 

"""

import ares

pars = \
{
 'fitting_function': 'ST',
 'Tmin': 1e4,
 'first_light_redshift': 40,
 'initial_redshift': 40,
 'cX': 3.4e40,
 'is_ion_src_igm': False,
 'is_ion_src_cgm': True,
 'fX': 0.2,
 'fXh': 0.2,
 'fesc': 0.1,
 'mu': 0.61,
 'clumping_factor': 3.,
}

sim = ares.simulations.Global21cm(**pars)
sim.run()
                    
anl = ares.analysis.Global21cm(sim)
ax = anl.GlobalSignature()
