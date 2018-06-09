"""

generate_hmf_tables.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed May  8 11:33:48 2013

Description: Create lookup tables for collapsed fraction. Can be run in 
parallel, e.g.,

    mpirun -np 4 python generate_hmf_tables.py

"""

import ares
import numpy as np

## INPUT
fit = 'PS'
format = 'npz'
##

pars = \
{
 "hmf_model": fit,
 # Should add halo concentration model here.
 "hmf_dlogM": 0.01,
 "hmf_logMmax": 18,
 "hmf_zmin": 3,
 "hmf_zmax": 60,
 "hmf_dz": 0.05,
 
 'mpowspec_dlnk': 0.05,
 'mpowspec_dlnR': 0.05,
 'mpowspec_lnk_min': -8.,
 'mpowspec_lnk_max': 8.,
 'mpowspec_lnR_min': -8.,
 'mpowspec_lnR_max': 8.,
}
##

hmf = ares.physics.HaloModel.HaloModel(hmf_load=True, hmf_load_ps=False, 
    **pars)

hmf.SavePS(format=format, clobber=False, checkpoint=True)



