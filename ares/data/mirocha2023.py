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
 'tau_approx': 0,#'neutral',
 'tau_clumpy': 0,#'madau1995',

 'cosmology_id': 'best',
 'cosmology_name': 'planck_TTTEEE_lowl_lowE',

 'final_redshift': 0,

 'tau_redshift_bins': 1000,

 'halo_dlnk': 0.05,
 'halo_lnk_min': -9.,
 'halo_lnk_max': 11.,
}

centrals_sf = \
{
 'pop_sfr_model': 'smhm-func',
 'pop_solve_rte': (0.12, E_LyA),
 'pop_Emin': 0.12,
 'pop_Emax': 24.6,

 'pop_centrals': True,
 'pop_zdead': 0,
 'pop_include_1h': False,
 'pop_include_2h': True,
 'pop_include_shot': True,

 # SED info
 'pop_sed': 'eldridge2017',
 'pop_rad_yield': 'from_sed',

 'pop_fesc': 0.2,
 'pop_sed_degrade': 10,

 'pop_nebular': 0,

 'pop_sfh': 'constant+ssp',
 'pop_ssp': (False, True),
 'pop_age': (100, 3e3),
 'pop_Z': (0.008, 0.02), # placeholder, really
 'pop_binaries': False,

 # Something with dust and metallicity here

 # fstar is SMHM for 'smhm-func' SFR model
 'pop_fstar': 'pq[0]',
 'pq_func[0]': 'dpl_evolNPS',
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
 'pq_func_par10[0]': 0.0,
 'pq_func_par11[0]': 0.0,
 'pq_func_par12[0]': 0.0,
 'pq_func_par13[0]': 0.0,

 'pq_val_ceil[0]': 1,

# sSFR(z, Mstell)
 'pop_ssfr': 'pq[1]',
 'pq_func[1]': 'dpl_evolNPS',
 'pq_func_var[1]': 'Ms',
 'pq_func_var2[1]': '1+z',
 'pq_func_par0[1]': 3e-10,
 'pq_func_par1[1]': 2e9,
 'pq_func_par2[1]': 0.0,
 'pq_func_par3[1]': -0.9,
 'pq_func_par4[1]': 1e9,
 'pq_func_par5[1]': 1.,
 'pq_func_par6[1]': 2.,     # PL index for normalization evolution
 'pq_func_par7[1]': 2.5,    # PL index for peak mass
 'pq_func_par8[1]': 0., # Only use if slopes evolve, e.g., in dplp_evolNPS
 'pq_func_par9[1]': 0., # Only use if slopes evolve, e.g., in dplp_evolNPS

 # Some occupation function stuff here.
 'pop_focc': 'pq[2]',
 'pq_func[2]': 'logtanh_abs_evolMFCW', # Evolving midpoint, floor, ceiling
 'pq_func_var[2]': 'Mh',
 'pq_func_var2[2]': '1+z',
 'pq_val_ceil[2]': 1,
 'pq_func_par0[2]': 1,
 'pq_func_par1[2]': 0.0,
 'pq_func_par2[2]': 11.8,
 'pq_func_par3[2]': 0.6,
 'pq_func_par4[2]': 1,   # redshift pivot
 'pq_func_par5[2]': 0.,
 'pq_func_par6[2]': 0,
 'pq_func_par7[2]': 0,
 'pq_func_par8[2]': 0,
 'pq_func_par9[2]': 0.,
 'pq_func_par10[2]': 0,
 'pq_func_par11[2]': 0,
 'pq_func_par12[2]': 0,
}

#centrals_sf.update(_base)

centrals_q = centrals_sf.copy()
centrals_q['pop_sfh'] = 'ssp'
centrals_q['pop_ssfr'] = None
centrals_q['pop_ssp'] = True
centrals_q['pop_age'] = 3e3
centrals_q['pop_Z'] = 0.02
centrals_q['pop_fstar'] = 'link:fstar:0'
centrals_q['pop_focc'] = 'link:focc:0'
centrals_q['pop_nebular'] = 0
centrals_q['pop_focc_inv'] = True

ihl_scaled = centrals_q.copy()
ihl_scaled['pop_focc'] = 1
ihl_scaled['pop_ihl'] = 0.01
#ihl_q['pq_func[0]'] = 'pl_evolN'
#ihl_q['pq_func_var[0]'] = 'Mh'
#ihl_q['pq_func_var2[0]'] = '1+z'
#ihl_q['pq_func_par0[0]'] = 0.01 # 1% of stellar mass -> IHL
#ihl_q['pq_func_par1[0]'] = 1e12
#ihl_q['pq_func_par2[0]'] = 0.  # Flat Mh dependence
#ihl_q['pq_func_par3[0]'] = 1.  # Anchored to z=0
#ihl_q['pq_func_par4[0]'] = 0.  # no evolution yet.
#ihl_q['pop_fstar'] = 'link:fstar:0'

ihl_scaled['pop_include_1h'] = True
ihl_scaled['pop_include_2h'] = True
ihl_scaled['pop_include_shot'] = False
ihl_scaled['pop_Mmin'] = 1e10
ihl_scaled['pop_Mmax'] = 1e14
ihl_scaled['pop_Tmin'] = None

satellites_sf = centrals_sf.copy()
satellites_sf['pop_focc'] = 1
satellites_sf['pop_focc_inv'] = False
satellites_sf['pop_centrals'] = 0
satellites_sf['pop_centrals_id'] = 0
satellites_sf['pop_prof_1h'] = 'nfw'
satellites_sf['pop_include_1h'] = True
satellites_sf['pop_include_2h'] = True
satellites_sf['pop_include_shot'] = False
satellites_sf['pop_fstar'] = 'link:fstar:0'
satellites_sf['pop_ssfr'] = 'link:ssfr:0'

satellites_q = centrals_q.copy()
satellites_q['pop_focc'] = 1
satellites_q['pop_focc_inv'] = False
satellites_q['pop_centrals'] = 0
satellites_q['pop_centrals_id'] = 2
satellites_q['pop_prof_1h'] = 'nfw'
satellites_q['pop_include_1h'] = True
satellites_q['pop_include_2h'] = True
satellites_q['pop_include_shot'] = False
satellites_q['pop_fstar'] = 'link:fstar:0'
satellites_q['pop_ssfr'] = None

#
#ihl_from_sat = centrals_sf_old.copy()
#ihl_from_sat['pop_focc'] = 1
#ihl_from_sat['pop_centrals'] = 0
#ihl_from_sat['pop_centrals_id'] = 0
#ihl_from_sat['pop_prof_1h'] = 'nfw'
#ihl_from_sat['pop_fsurv'] = 'link:fsurv:3'
#ihl_from_sat['pop_surv_inv'] = True
#ihl_from_sat['pop_include_1h'] = True
#ihl_from_sat['pop_include_2h'] = True
#ihl_from_sat['pop_include_shot'] = False

base = _base.copy()
_pop0 = centrals_sf.copy()
_pop1 = centrals_q.copy()
_pop2 = satellites_sf.copy()
_pop3 = satellites_q.copy()

for i, _pop in enumerate([_pop0, _pop1]):
    pf = {}
    for par in _pop.keys():
        pf[par + '{%i}' % i] = _pop[par]

    base.update(pf)

subhalos = {}
for i, _pop in enumerate([_pop2, _pop3]):
    pf = {}
    for par in _pop.keys():
        pf[par + '{%i}' % (i + 2)] = _pop[par]

    subhalos.update(pf)

subhalos['pop_fsurv{2}'] = 'pq[3]'
subhalos['pq_func[3]{2}'] = 'logtanh_abs_evolM'
subhalos['pq_func_var[3]{2}'] = 'Mh'
subhalos['pq_func_var2[3]{2}'] = '1+z'
subhalos['pq_val_ceil[3]{2}'] = 1
subhalos['pq_func_par0[3]{2}'] = 0.0  # step = par0-par1
subhalos['pq_func_par1[3]{2}'] = 0.95 # fsurv = par1 + step * tanh(stuff)
subhalos['pq_func_par2[3]{2}'] = 11
subhalos['pq_func_par3[3]{2}'] = 0.7 # dlogM
subhalos['pq_func_par4[3]{2}'] = 0.  # Evolution in midpoint
subhalos['pq_func_par5[3]{2}'] = 1   # Pin to z=0

subhalos['pop_fsurv{3}'] = 'link:fsurv:2'
subhalos['pop_fsurv_inv{3}'] = True

dust = {}
dust['pop_dust_template'] = 'WD01:MWRV31'
dust['pop_Av'] = 'pq[4]'
dust['pq_func[4]'] = 'pl_evolN'
dust['pq_func_var[4]'] = 'Ms'
dust['pq_func_var2[4]'] = '1+z'
dust['pq_func_par0[4]'] = 0.3
dust['pq_func_par1[4]'] = 1e10
dust['pq_func_par2[4]'] = 0.1
dust['pq_func_par3[4]'] = 1.  # Anchored to z=0
dust['pq_func_par4[4]'] = 0.  # no evolution yet.

mzr = \
{
 'pop_enrichment': True,
 'pop_mzr': 'pq[30]',
 'pop_fox': 0.03,
 "pq_func[30]": 'linear_evolN',
 'pq_func_var[30]': 'Ms',
 'pq_func_var2[30]': '1+z',
 'pq_func_par0[30]': 8.65,
 'pq_func_par1[30]': 10,
 'pq_func_par2[30]': 0.2,
 'pq_func_par3[30]': 1.,     # pin to z=0
 'pq_func_par4[30]': -0.08,   # mild evolution
 'pq_val_ceil[30]': 9,
 'pq_val_floor[30]': 6,
 'pop_Z': ('mzr', 0.02),
}


expd = base.copy()
expd['pop_ssp{0}'] = True
expd['pop_sfh{0}'] = 'exp_decl'
expd['pop_sfh_axes{0}'] = ('norm', 10**np.arange(-6, 4.2, 0.2)), \
                          ('tau', 10**np.arange(2, 4.2, 0.2))

expd['pop_age{0}'] = None
expd['pop_Z{0}'] = 0.02
expd['pop_aging{0}'] = True
expd['pop_sed_degrade{0}'] = 10
expd['pop_sfh_degrade{0}'] = 1  # Tabulate in (z, Mh) degraded by 10x wrt native
