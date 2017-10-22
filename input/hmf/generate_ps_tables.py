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
fit = 'ST'
format = 'npz'
##

hmf_pars = \
{
 "hmf_model": fit,
 "hmf_dlogM": 0.01,
 "hmf_zmin": 3.,
 "hmf_dz": 0.05,
 "hmf_zmax": 60.,
 'fft_scales': np.arange(1e-4, 1e2+1e-4, 1e-4),
 'powspec_dlogk': 0.05,
 'powspec_dlogr': 0.05,
}
##

hmf = ares.physics.HaloModel(hmf_load=True, hmf_load_ps=False, **hmf_pars)

hmf.save_ps(format=format, clobber=True)



