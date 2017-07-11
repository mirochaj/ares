"""

test_parallel_grid.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Fri Jul  7 15:02:45 PDT 2017

Description: 

"""

import sys
import ares
import numpy as np
from mpi4py import MPI

rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size


prefix = sys.argv[1]
exit = int(sys.argv[2])

blobs = ares.util.BlobBundle('gs:basics')

base_kwargs = {'tanh_model': True, 'problem_type': 101}
base_kwargs.update(blobs)

mg = ares.inference.ModelGrid(**base_kwargs)

z0 = np.arange(6, 13, 1)
dz = np.arange(1, 9, 1)

mg.axes = {'tanh_xz0': z0, 'tanh_xdz': dz}

mg.checkpoint_by_proc = True
mg.LoadBalance()

freq = int((mg.grid.size * 0.5) / size)

mg.run(prefix, clobber=False, restart=True, save_freq=freq, exit_after=exit)


