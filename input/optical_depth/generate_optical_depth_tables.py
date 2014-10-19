"""

generate_optical_depth_tables.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri Jun 14 09:20:22 2013

Description: 

Note: This can be run in parallel, e.g.,

    mpirun -np 4 python generate_optical_depth_tables.py

"""

import numpy as np
import os, glorb, h5py, time
from rt1d.physics.Constants import E_LL

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1

#
## INPUT
zf, zi = (10, 40)
Emin = 1e2
Emax = 5e4
Nz = [400, 600, 800]
format = 'hdf5'        # 'hdf5' or 'txt'
approx_helium = 0
##
#

# Initialize radiation background
pars = \
{
 'spectrum_Emin': Emin,
 'spectrum_Emax': Emax,
 'approx_xray': 0,
 'xray_cutoff': Emin,
 'approx_helium': approx_helium,
 'initial_redshift': zi,
 'final_redshift': zf,
}

# Track how long it takes, output result to file
if rank == 0 and not os.path.exists('tau_tab_timing.txt'):
    t = open('tau_tab_timing.txt', 'w')
    print >> t, "#                fn            time (s)    # of cores"
    t.close()

for res in Nz:

    pars.update({'redshift_bins': res})

    # Start timer
    t1 = time.time()
    
    # Create IGM instance
    igm = glorb.evolve.IGM(**pars)
    
    fn = igm.tau_name(suffix=format)[0]
    
    if os.path.exists(fn):
        if rank == 0:
            raise IOError('%s exists! Exiting.' % fn)
    
    if rank == 0:
        print "Now creating %s..." % fn
    
    tau = igm.TabulateOpticalDepth()
    
    t2 = time.time()

    if rank != 0:
        continue
    
    if format == 'hdf5':
        f = h5py.File(fn, 'w')
        f.create_dataset('tau', data=tau)
        f.create_dataset('redshift', data=igm.z)
        f.create_dataset('photon_energy', data=igm.E)
        f.close()
    else:
        f = open(fn, 'w')
        hdr = "zmin=%.4g zmax=%.4g Emin=%.8e Emax=%.8e" % \
            (igm.z.min(), igm.z.max(), igm.E.min(), igm.E.max())
        np.savetxt(fn, tau, header=hdr, fmt='%.8e')

    print 'Wrote %s.' % fn
    t = open('tau_tab_timing.txt', 'a')
    print >> t, "%s %g %i" % (fn, (t2-t1), size)
    t.close()
    
    