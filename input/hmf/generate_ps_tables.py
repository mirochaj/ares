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
fmt = 'hdf5'
##

pars = \
{
 "hmf_model": fit,
 # Should add halo concentration model here.
 "hmf_dlogM": 0.01,
 "hmf_logMmin": 4,
 "hmf_logMmax": 18,
 "hmf_zmin": 5,
 "hmf_zmax": 30,
 "hmf_dz": 0.05,

 "hps_zmin": 6,
 "hps_zmax": 30,
 "hps_dz": 0.5,

 'hps_dlnk': 0.001,
 'hps_dlnR': 0.001,
 'hps_lnk_min': -10.,
 'hps_lnk_max': 10.,
 'hps_lnR_min': -10.,
 'hps_lnR_max': 10.,
}

kwargs = \
{
 'split_by_scale': True,
 'epsrel': 1e-8,
 'epsabs': 1e-8,
}

##

hmf = ares.physics.HaloModel(hmf_load=True, hmf_load_ps=False,
    **pars)

hmf.SavePS(format=fmt, clobber=False, checkpoint=True, **kwargs)
