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
fit = 'ST'
format = 'npz'
##


hmf_pars = \
{
 "hmf_model": fit,
 "hmf_dlogM": 0.01,
 "hmf_zmin": 3.,
 "hmf_dz": 0.05,
}
##

hmf = ares.physics.HaloMassFunction(hmf_analytic=False, 
    hmf_load=False, **hmf_pars)

hmf.save(format=format)


