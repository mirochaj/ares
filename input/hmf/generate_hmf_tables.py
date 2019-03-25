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

## INPUT
fit = 'Tinker10'
fmt = 'npz'
##

hmf_pars = \
{
 "hmf_model": fit,
 "hmf_zmin": 0.,
 "hmf_zmax": 60.,
 "hmf_dz": 0.05,
 "hmf_logMmin": 4,
 "hmf_logMmax": 18,
 "hmf_dlogM": 0.01,
 "hmf_dt": 1.,
 "hmf_tmax": 2000.,
}
##

hmf = ares.physics.HaloMassFunction(hmf_analytic=False, 
    hmf_load=False, **hmf_pars)

hmf.SaveHMF(format=fmt)


