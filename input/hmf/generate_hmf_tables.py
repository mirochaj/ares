"""

generate_hmf_tables.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed May  8 11:33:48 2013

Description: Create lookup tables for collapsed fraction. Can be run in
parallel, e.g.,

    mpirun -np 4 python generate_hmf_tables.py

"""

import sys
import ares

def_kwargs = \
{
 "hmf_model": 'Tinker10',
 "hmf_logMmin": 4,
 "hmf_logMmax": 18,
 "hmf_dlogM": 0.01,

 "hmf_fmt": 'hdf5',
 "hmf_table": None,
 "hmf_wdm_mass": None,

 #"hmf_window": 'sharpk',

 # Redshift sampling
 #"hmf_zmin": 0.,
 #"hmf_zmax": 60.,
 #"hmf_dz": 0.05,

 # Can do constant timestep instead of constant dz
 "hmf_dt": 10,
 "hmf_tmin": 30.,
 "hmf_tmax": 13.7e3, # Myr

 # Cosmology
 "cosmology_id": 'best',
 "cosmology_name": 'planck_TTTEEE_lowl_lowE',

 #HMF params and filter params are for doing Aurel Schneider's 2015 paper WDM.
 #"hmf_params" : {'a' : 1.0},
 #"filter_params" : {'c' : 2.5}

 #"cosmology_id": 'paul',
 #"cosmology_name": 'user',
 #"sigma_8": 0.8159,
 #'primordial_index': 0.9652,
 #'omega_m_0': 0.315579,
 #'omega_b_0': 0.0491,
 #'hubble_0': 0.6726,
 #'omega_l_0': 1. - 0.315579,

}

##

kwargs = def_kwargs.copy()
kwargs.update(ares.util.get_cmd_line_kwargs(sys.argv))

hmf = ares.physics.HaloMassFunction(hmf_analytic=False,
    hmf_load=False, **kwargs)

hmf.info

try:
    hmf.SaveHMF(fmt=kwargs['hmf_fmt'], clobber=False)
except IOError as err:
    print(err)
