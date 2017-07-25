"""

generate_simpl_seds.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Thu May 11 10:03:29 CDT 2017

Description: 

"""

import os
import ares
import time
import numpy as np
from mpi4py import MPI

rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size

#
## INPUT
mass = 10.
E = 10**np.arange(1, 5.1, 0.1)
##
#

simpl = \
{
 'source_type': 'bh', 
 'source_mass': mass,
 'source_rmax': 1e2,
 'source_sed': 'simpl',
 'source_Emin': 1,
 'source_Emax': 5e4,
 'source_EminNorm': 500.,
 'source_EmaxNorm': 8e3,
 'source_alpha': -1.5,
 'source_fsc': 0.1,
 'source_dlogE': 0.025,
}

for i, alpha in enumerate([-2.5, -2, -1.5, -1, -0.5, -0.25]):
    for j, fsc in enumerate([0.1, 0.5, 0.9]):
        
        k = i * 3 + j
        
        if k % size != rank:
            continue
        
        fn = 'simpl_M_%i_fsc_%.2f_alpha_%.2f.txt' % (mass, fsc, alpha)
        
        if os.path.exists(fn):
            print "%s already exists." % fn
            continue
        
        simpl['source_alpha'] = alpha
        simpl['source_fsc'] = fsc
        
        src = ares.sources.BlackHole(**simpl)
        src.dump(fn, E)
        
