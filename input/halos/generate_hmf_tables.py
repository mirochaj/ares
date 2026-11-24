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
 "halo_mf": 'Tinker10',
 "halo_logMmin": 4,
 "halo_logMmax": 18,
 "halo_dlogM": 0.01,

 "halo_fmt": 'hdf5',
 "halo_table": None,
 "halo_wdm_mass": None,

 #"halo_window": 'sharpk',

 # Redshift sampling
 "halo_zmin": 0.,
 "halo_zmax": 60.,
 "halo_dz": 0.05,

 # Can do constant timestep instead of constant dz
 #"halo_dt": 100,
 #"halo_tmin": 100.,
 #"halo_tmax": 13.7e3, # Myr

 # Cosmology
 "cosmology_id": 'best',
 "cosmology_name": 'planck_TTTEEE_lowl_lowE',

 #HMF params and filter params are for doing Aurel Schneider's 2015 paper WDM.
 #"halo_params" : {'a' : 1.0},
 #"filter_params" : {'c' : 2.5}

}

##

kwargs = def_kwargs.copy()
kwargs.update(ares.util.get_cmd_line_kwargs(sys.argv))

halos = ares.physics.HaloMassFunction(halo_mf_analytic=False,
    halo_mf_load=False, **kwargs)

halos.info

try:
    halos.save_hmf(fmt='hdf5', clobber=False)
except IOError as err:
    print(err)
