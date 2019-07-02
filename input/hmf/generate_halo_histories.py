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
    fn_hmf = 'hmf_ST_logM_1400_4-18_z_1201_0-60.npz'

pars = ares.util.ParameterBundle('mirocha2017:base').pars_by_pop(0, 1) \
     + ares.util.ParameterBundle('mirocha2017:dflex').pars_by_pop(0, 1)

pars['hmf_table'] = fn_hmf

cosmo = \
{
 "sigma_8": 0.8159, 
 'primordial_index': 0.9652, 
 'omega_m_0': 0.315579, 
 'omega_b_0': 0.0491, 
 'hubble_0': 0.6726,
 'omega_l_0': 1. - 0.315579,
}

pars.update(cosmo)

pop = ares.populations.GalaxyPopulation(**pars)

if 'npz' in fn_hmf:
    pref = fn_hmf.replace('.npz', '')
elif 'hdf5' in fn_hmf:
    pref = fn_hmf.replace('.hdf5', '')
else:
    raise IOError('Unrecognized file format for HMF ({})'.format(fn_hmf))
    
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
    
    # Save cosmology
    grp = f.create_group('cosmology')
    for key in cosmo:
        grp.create_dataset(key, data=cosmo[key])
        
    f.close()    
    print("Wrote {}".format(fn))