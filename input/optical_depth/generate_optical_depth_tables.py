"""

generate_optical_depth_tables.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri Jun 14 09:20:22 2013

Description: Generate optical depth lookup table.

Note: This can be run in parallel, e.g.,

    mpirun -np 4 python generate_optical_depth_tables.py

"""

import numpy as np
import os, ares, time, pickle

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
zf, zi = (10, 50)
Emin = 1e2
Emax = 5e4
Nz = [1e3]
format = 'hdf5'        # 'hdf5' or 'pkl' or 'npz'
helium = 1
##
#

# Initialize radiation background
pars = \
{
 'include_He': helium,
 'spectrum_Emin': Emin,
 'spectrum_Emax': Emax,
 'approx_xrb': 0,
 'approx_He': helium,
 'initial_redshift': zi,
 'final_redshift': zf,
}

# Track how long it takes, output result to file
#if rank == 0 and not os.path.exists('tau_tab_timing.txt'):
#    t = open('tau_tab_timing.txt', 'w')
#    print >> t, "#                fn            time (s)    # of cores"
#    t.close()

for res in Nz:

    pars.update({'redshift_bins': res})

    # Start timer
    t1 = time.time()
    
    # Create IGM instance
    igm = ares.solvers.IGM(**pars)
    
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
    elif format == 'npz':
        to_write = {'tau': tau, 'z': igm.z, 'E': igm.E}
        
        f = open(fn, 'w')
        np.savez(f, **to_write)
        f.close()
        
    elif format == 'pkl':
        
        f = open(fn, 'wb')
        pickle.dump({'tau': tau, 'z': igm.z, 'E': igm.E}, f)
        f.close()    
        
    else:
        f = open(fn, 'w')
        hdr = "zmin=%.4g zmax=%.4g Emin=%.8e Emax=%.8e" % \
            (igm.z.min(), igm.z.max(), igm.E.min(), igm.E.max())
        np.savetxt(fn, tau, header=hdr, fmt='%.8e')

    print 'Wrote %s.' % fn
    #t = open('tau_tab_timing.txt', 'a')
    #print >> t, "%s %g %i" % (fn, (t2-t1), size)
    #t.close()
    
    