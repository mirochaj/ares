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
seed = int(sys.argv[2])

blobs = ares.util.BlobBundle('gs:basics')
base_kwargs = {'tanh_model': True, 'problem_type': 101}
base_kwargs.update(blobs)

ps = ares.inference.PriorSet()
ps.add_prior(ares.inference.Priors.UniformPrior(0, 3), 'tanh_T0')
ps.add_prior(ares.inference.Priors.UniformPrior(6, 20.), 'tanh_Tz0')
ps.add_prior(ares.inference.Priors.UniformPrior(0.1, 10.), 'tanh_Tdz')

mc = ares.inference.ModelSample(**base_kwargs)

mc.prior_set = ps
mc.N = 1e2            # Number of models to run
mc.save_by_proc = True
mc.is_log = {'tanh_T0': True}
mc.seed = seed       
            
mc.run(prefix, clobber=True, restart=False)

# Should also test restart etc.
# If seed is provided, restart should run *up to* N models, not N more.


