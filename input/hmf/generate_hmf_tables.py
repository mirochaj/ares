"""

generate_hmf_tables.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed May  8 11:33:48 2013

Description: Create lookup tables for collapsed fraction. Can be run in 
parallel, e.g.,

    mpirun -np 4 python generate_hmf_tables.py

"""

import os, ares

## INPUT
fit = 'ST'
format = 'pkl'
##


hmf_pars = \
{
 "hmf_func": fit,
 "hmf_dlogM": 0.01,
 "hmf_zmin": 4.,
 "hmf_dz": 0.01,
}
##

hmf = ares.physics.HaloMassFunction(hmf_analytic=False, 
    hmf_load=False, **hmf_pars)

hmf.save(format=format)


