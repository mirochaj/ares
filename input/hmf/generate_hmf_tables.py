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
format = 'pkl'

hmf_pars = \
{
 "fitting_function": 'ST',
 "hmf_logMmin": 4,
 "hmf_logMmax": 16,
 "hmf_dlogM": 0.05,
 "hmf_zmin": 4,
 "hmf_zmax": 80,
 "hmf_dz": 0.05,    
}
##

hmf = ares.populations.HaloMassFunction.HaloDensity(hmf_analytic=False, 
    load_hmf=False, **hmf_pars)

# Write to $GLORB/input directory
if not os.path.exists('../input'):
    os.mkdir('../input')
    if not os.path.exists('../input/hmf'):
        os.mkdir('../input/hmf')

hmf.save(destination='../input/hmf', format=format)


