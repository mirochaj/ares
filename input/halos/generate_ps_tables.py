"""

generate_ps_tables.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed May  8 11:33:48 2013

Description: Create lookup tables for collapsed fraction. Can be run in
parallel, e.g.,

    mpirun -np 4 python generate_ps_tables.py

"""

import ares
import numpy as np

## INPUT
fit = 'ST'
fmt = 'hdf5'
##

pars = \
{
 "halo_mf": fit,
 # Should add halo concentration model here.
 "halo_dlogM": 0.01,
 "halo_logMmin": 4,
 "halo_logMmax": 18,
 "halo_zmin": 0,
 "halo_zmax": 60,
 "halo_dz": 0.05,

 'halo_dlnk': 0.001,
 'halo_dlnR': 0.001,
 'halo_lnk_min': -9.,
 'halo_lnk_max': 9.,
 'halo_lnR_min': -9.,
 'halo_lnR_max': 9.,
}

kwargs = \
{
 'split_by_scale': True,
 'epsrel': 1e-8,
 'epsabs': 1e-8,
}

##

halos = ares.physics.HaloModel(halo_mf_load=True, halos_ps_load=False,
    **pars)

halos.generate_ps(format=fmt, clobber=False, checkpoint=True, **kwargs)
