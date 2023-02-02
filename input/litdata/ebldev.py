import numpy as np
from ares.physics.Constants import E_LyA, E_LL

_base = \
{
 "halo_dt": 10,
 "halo_tmin": 30.,
 "halo_tmax": 13.7e3, # Myr
 'halo_mf': 'Tinker10',
 "halo_mf_sub": 'Tinker08',

 # NIRB
 'tau_approx': 'neutral',
 'tau_clumpy': 'madau1995',

 'cosmology_id': 'best',
 'cosmology_name': 'planck_TTTEEE_lowl_lowE',

 'final_redshift': 0,

 'pop_Emin': 0.25, # about 5 microns
 'pop_solve_rte': (0.25, E_LL),
 'tau_clumpy': 0.,
 'tau_redshift_bins': 1000,

 'halo_dlnk': 0.05,
 'halo_lnk_min': -9.,
 'halo_lnk_max': 11.,
}


centrals_sf = \
{
 'pop_sfr_model': 'smhm-func',

 'pop_centrals': True,
 'pop_zdead': 0,
 'pop_include_1h': False,
 'pop_include_2h': True,
 'pop_include_shot': True,

 # SED info
 'pop_sed': 'eldridge2009',
 'pop_binaries': False,
 'pop_rad_yield': 'from_sed',
 'pop_Emin': E_LyA,
 'pop_Emax': 24.6,
 'pop_fesc': 0.2,
 'pop_sed_degrade': 10,

 # Something with dust and metallicity here

 # fstar is SMHM for 'smhm-func' SFR model
 'pop_fstar': 'pq[0]',
 'pq_func[0]': 'dpl_evolNP',
 'pq_func_var[0]': 'Mh',
 'pq_func_var2[0]': '1+z',
 'pq_func_par0[0]': 4e-4,
 'pq_func_par1[0]': 1.5e12,
 'pq_func_par2[0]': 1.0,
 'pq_func_par3[0]': -0.6,
 'pq_func_par4[0]': 1e10,
 'pq_func_par5[0]': 1.,     # pivot in 1+z
 'pq_func_par6[0]': 0.,     # norm
 'pq_func_par7[0]': 0.0,    # Mp
 'pq_func_par8[0]': 0.0,    # Only use if slopes evolve, e.g., in dplp_evolNPS
 'pq_func_par9[0]': 0.0,    # Only use if slopes evolve, e.g., in dplp_evolNPS
 'pq_val_ceil[0]': 1,

# sSFR(z, Mstell)
 'pop_ssfr': 'pq[1]',
 #'pq_func[1]': 'pl_evolN',
 #'pq_func_var[1]': 'Ms',
 #'pq_func_var2[1]': '1+z',
 #'pq_func_par0[1]': 2e-10,
 #'pq_func_par1[1]': 1e10,
 #'pq_func_par2[1]': -0.0,
 #'pq_func_par3[1]': 1.,
 #'pq_func_par4[1]': 1.5,

 # Add mass evolution too?
 'pq_func[1]': 'dpl_evolNP',
 'pq_func_var[1]': 'Ms',
 'pq_func_var2[1]': '1+z',
 'pq_func_par0[1]': 4e-10,
 'pq_func_par1[1]': 5e9,
 'pq_func_par2[1]': 0.0,
 'pq_func_par3[1]': -0.9,
 'pq_func_par4[1]': 1e9,
 'pq_func_par5[1]': 1.,
 'pq_func_par6[1]': 0., # Redshift evol
 'pq_func_par7[1]': 0., # peak evol

 # Some occupation function stuff here.
 'pop_focc': 'pq[2]',
 #'pq_func[2]': 'pl_evolN',
 #'pq_func_var[2]': 'Mh',
 #'pq_func_var2[2]': '1+z',
 #'pq_val_ceil[2]': 1,
 #'pq_func_par0[2]': 0.5,
 #'pq_func_par1[2]': 1e11,
 #'pq_func_par2[2]': -0.5,
 #'pq_func_par3[2]': 0.5,
 #'pq_func_par4[2]': 1,

 'pq_func[2]': 'logtanh_abs_evolM',
 'pq_func_var[2]': 'Mh',
 'pq_func_var2[2]': '1+z',
 'pq_val_ceil[2]': 1,
 'pq_func_par0[2]': 0.95,
 'pq_func_par1[2]': 0.0,
 'pq_func_par2[2]': 11.8,
 'pq_func_par3[2]': 0.6,
 'pq_func_par4[2]': 0.1,
 'pq_func_par5[2]': 1,

}

centrals_sf.update(_base)

centrals_q = centrals_sf.copy()
centrals_q['pop_ssfr'] = None
centrals_q['pop_ssp'] = True
centrals_q['pop_age'] = 3e3
centrals_q['pop_Z'] = 0.02

ihl_q = centrals_q.copy()
ihl_q['pop_ihl'] = 1
ihl_q['pq_func[0]'] = 'pl_evolN'
ihl_q['pq_func_var[0]'] = 'Mh'
ihl_q['pq_func_var2[0]'] = '1+z'
ihl_q['pq_func_par0[0]'] *= 0.01 # 1% of stellar mass -> IHL
#ihl_q['pq_func_par1[0]'] = 1e12
#ihl_q['pq_func_par2[0]'] = 0.  # Flat Mh dependence
#ihl_q['pq_func_par3[0]'] = 1.
#ihl_q['pq_func_par4[0]'] = 0.  # no evolution yet.

ihl_q['pop_Z'] = 0.02
ihl_q['pop_include_1h'] = True
ihl_q['pop_include_2h'] = True
ihl_q['pop_include_shot'] = False
ihl_q['pop_Mmin'] = 1e10
ihl_q['pop_Mmax'] = 1e14
ihl_q['pop_Tmin'] = None

satellites_sf = centrals_sf.copy()
satellites_sf['pop_centrals'] = 0
satellites_sf['pop_centrals_id'] = 0
satellites_sf['pop_prof_1h'] = 'nfw'
satellites_sf['pop_include_1h'] = True
satellites_sf['pop_include_2h'] = True
satellites_sf['pop_include_shot'] = False
