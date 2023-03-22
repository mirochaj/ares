"""

generate_prof_tables.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed May  8 11:33:48 2013

Description: Create lookup tables for Fourier-transformed profiles.

"""

import ares
import numpy as np

## INPUT
fit = 'Tinker10'
fmt = 'hdf5'
##

pars = \
{
 "halo_mf": fit,

 # Should add halo concentration model here.
 "halo_dlogM": 0.01,
 "halo_logMmin": 4,
 "halo_logMmax": 18,
 #"halo_zmin": 0,
 #"halo_zmax": 60,
 #"halo_dz": 0.05,

 #"hps_zmin": 0,
 #"hps_zmax": 30,
 #"hps_dz": 0.1,

 "halo_dt": 10,
 "halo_tmin": 30.,
 "halo_tmax": 13.7e3, # Myr


 'halo_dlnk': 0.05,
 'halo_dlnR': 0.001,
 'halo_lnk_min': -9.,
 'halo_lnk_max': 11.,
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

halos = ares.physics.HaloModel(halo_mf_load=True, halo_ps_load=False,
    **pars)

halos.generate_halo_prof(format=fmt, clobber=False, checkpoint=True, **kwargs)