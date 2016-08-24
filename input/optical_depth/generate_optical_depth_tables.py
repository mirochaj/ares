"""

generate_optical_depth_tables.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri Jun 14 09:20:22 2013

Description: Generate optical depth lookup table.

Note: This can be run in parallel, e.g.,

    mpirun -np 4 python generate_optical_depth_tables.py

"""

import ares
import numpy as np
import os, time, pickle
from ares.physics.Constants import E_LL, E_LyA

try:
    import h5py
except:
    pass

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1

#
## INPUT
zf, zi = (5, 50)
Emin = 2e2
Emax = 3e4
Nz = [400]
format = 'npz'        # 'hdf5' or 'pkl' or 'npz'
helium = 1
xavg = lambda z: 0.0  # neutral
##
#

# Initialize radiation background
pars = \
{
 'include_He': helium,
 'pop_Emin': Emin,
 'pop_Emax': Emax,
 'approx_He': helium,
 'initial_redshift': zi,
 'final_redshift': zf,
}

for res in Nz:

    pars.update({'pop_tau_Nz': res})

    # Create OpticalDepth instance
    igm = ares.solvers.OpticalDepth(**pars)
    
    # Impose an ionization history: neutral for all times
    igm.ionization_history = xavg
    
    # Tabulate tau
    tau = igm.TabulateOpticalDepth()
    igm.save(suffix=format, clobber=True)

    