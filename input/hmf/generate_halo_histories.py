"""

run_trajectories.py

Author: Jordan Mirocha
Affiliation: McGill
Created on: Sat  9 Mar 2019 15:48:15 EST

Description: This script may be obsolete.

"""

import os
import sys
import ares
import h5py
import numpy as np
import matplotlib.pyplot as pl

try:
    fn_hmf = sys.argv[1]
except IndexError:
    fn_hmf = 'hmf_ST_planck_TTTEEE_lowl_lowE_best_logM_1400_4-18_z_1201_0-60.hdf5'

pars = ares.util.ParameterBundle('mirocha2017:base').pars_by_pop(0, 1) \
     + ares.util.ParameterBundle('mirocha2017:dflex').pars_by_pop(0, 1)

pars['hmf_table'] = fn_hmf

with h5py.File(fn_hmf, 'r') as f:
    grp = f['cosmology']
    
    cosmo_pars = {}
    cosmo_pars['cosmology_name'] = grp.attrs.get('cosmology_name')
    cosmo_pars['cosmology_id'] = grp.attrs.get('cosmology_id')
    
    for key in grp:
        buff = np.zeros(1)
        grp[key].read_direct(buff)
        cosmo_pars[key] = buff[0]
            
    print("Read cosmology from {}.".format(fn_hmf))

pars.update(cosmo_pars)

# We might periodically tinker with these things but these are good defaults.
pars['pop_Tmin'] = None
pars['pop_Mmin'] = 1e4
pars['hgh_dlogMmin'] = 0.1
pars['hgh_Mmax'] = 10

pop = ares.populations.GalaxyPopulation(**pars)

if 'npz' in fn_hmf:
    pref = fn_hmf.replace('.npz', '').replace('hmf', 'hgh')
elif 'hdf5' in fn_hmf:
    pref = fn_hmf.replace('.hdf5', '').replace('hmf', 'hgh')
else:
    raise IOError('Unrecognized file format for HMF ({})'.format(fn_hmf))
    
if pars['hgh_dlogMmin'] is not None:
    pref += '_xM_{:.0f}_{:.2f}'.format(pars['hgh_Mmax'], pars['hgh_dlogMmin'])
    
fn = '{}.hdf5'.format(pref)
if not os.path.exists(fn):

    print("Running new trajectories...")
    zall, hist = pop.Trajectories()

    f = h5py.File(fn, 'w')
    
    # Save halo trajectories
    for key in hist:
        if key not in ['z', 't', 'nh', 'Mh', 'MAR']:
            continue
        f.create_dataset(key, data=hist[key])
        
    f.close()    
    print("Wrote {}".format(fn))
    
else:
    print("File {} exists. Exiting.".format(fn))    