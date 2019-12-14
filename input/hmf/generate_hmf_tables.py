"""

generate_hmf_tables.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed May  8 11:33:48 2013

Description: Create lookup tables for collapsed fraction. Can be run in 
parallel, e.g.,

    mpirun -np 4 python generate_hmf_tables.py

"""

import sys
import ares

## INPUT
fit = 'ST'
fmt = 'npz'

hmf_pars = \
{
 "hmf_model": fit,
 "hmf_logMmin": 4,
 "hmf_logMmax": 18,
 "hmf_dlogM": 0.01,

 #"hmf_window": 'sharpk',

 # Redshift sampling
 "hmf_zmin": 0.,
 "hmf_zmax": 60.,
 "hmf_dz": 0.05, 
 
 # Can do constant timestep instead of constant dz 
 #"hmf_dt": 1.,
 #"hmf_tmin": 30.,
 #"hmf_tmax": 2000.,
 
 # Cosmology
 "sigma_8": 0.8159, 
 'primordial_index': 0.9652, 
 'omega_m_0': 0.315579, 
 'omega_b_0': 0.0491, 
 'hubble_0': 0.6726,
 'omega_l_0': 1. - 0.315579, 
}

##

hmf = ares.physics.HaloMassFunction(hmf_analytic=False, 
    hmf_load=False, **hmf_pars)

hmf.SaveHMF(format=fmt, clobber=False)


