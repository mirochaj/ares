"""

generate_surf_tables.py

Author: Jordan Mirocha
Affiliation: JPL / Caltech
Created on: Thu Feb  2 08:50:27 PST 2023

Description:

"""

import ares
import numpy as np
import matplotlib.pyplot as pl

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

halos = ares.physics.HaloModel(halo_mf_load=True, **pars)

halos.generate_halo_surface_dens(format=fmt, clobber=False, checkpoint=True)
