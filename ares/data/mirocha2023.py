import numpy as np
from ares.physics.Constants import E_LyA

_base = \
{
 "halo_dt": 10,
 "halo_tmin": 30.,
 "halo_tmax": 13.7e3, # Myr
 'halo_mf': 'Tinker10',
 "halo_mf_sub": 'Tinker08',

 # NIRB
 'tau_approx': 0,#'neutral',
 'tau_clumpy': 2,

 'cosmology_id': 'best',
 'cosmology_name': 'planck_TTTEEE_lowl_lowE',
 'cosmological_Mmin': None,

 'final_redshift': 1e-2,

 'tau_redshift_bins': 100,

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
 'pop_sed_degrade': 100,

 'pop_nebular': 0,

 'pop_sfh': 'constant+ssp',
 'pop_ssp': (False, True),
 'pop_age': (100, 1e4),
 'pop_Z': (0.02, 0.02), # placeholder, really
 'pop_binaries': False,

 'pop_Tmin': None,
 'pop_Mmin': 1e8,

 # Something with dust and metallicity here

 # fstar is SMHM for 'smhm-func' SFR model
 'pop_fstar': 'pq[0]',
 'pq_func[0]': 'dpl_evolB13',
 'pq_func_var[0]': 'Mh',
 'pq_func_var2[0]': '1+z',
 'pq_func_par0[0]': 0.0003,
 'pq_func_par1[0]': 1.5e12,
 'pq_func_par2[0]': 1,
 'pq_func_par3[0]': -0.6,
 'pq_func_par4[0]': 1e10,           # normalization pinned to this Mh
 'pq_func_par5[0]': 0,
 'pq_func_par6[0]': 0,
 'pq_func_par7[0]': 0,
 'pq_func_par8[0]': 0,
 'pq_func_par9[0]': 0.0,
 'pq_func_par10[0]': 0.0,
 'pq_func_par11[0]': 0.0,
 'pq_func_par12[0]': 0.0,
 'pq_func_par13[0]': 0.0,
 'pq_func_par14[0]': 0.0,
 'pq_func_par15[0]': 0.0,
 'pq_func_par16[0]': 0.0,

 'pq_val_ceil[0]': 1,

# sSFR(z, Mstell)
 'pop_ssfr': 'pq[1]',
 'pq_func[1]': 'dpl_evolB13',
 'pq_func_var[1]': 'Ms',
 'pq_func_var2[1]': '1+z',
 'pq_func_par0[1]': 5e-10,
 'pq_func_par1[1]': 2e9,
 'pq_func_par2[1]': -0.1,
 'pq_func_par3[1]': -0.7,
 'pq_func_par4[1]': 1e8,   # Mstell anchor
 'pq_func_par5[1]': 2.,    # scales (1-a) term
 'pq_func_par6[1]': 0.,    # scales (1-a) term
 'pq_func_par7[1]': 0,     # scales (1-a) term
 'pq_func_par8[1]': 0,     # scales (1-a) term
 'pq_func_par9[1]': 0.2,   # scales log(1+z) term
 'pq_func_par10[1]': 0.0,  # scales log(1+z) term
 'pq_func_par11[1]': 0.0,  # scales log(1+z) term
 'pq_func_par12[1]': 0.0,  # scales log(1+z) term
 'pq_func_par13[1]': 0.0,
 'pq_func_par14[1]': 0.0,
 'pq_func_par15[1]': 0.0,
 'pq_func_par16[1]': 0.0,

 # Some occupation function stuff here.
 'pop_focc': 'pq[2]',
 'pq_func[2]': 'erf_evolB13',#'logsigmoid_abs_evol_FCW', # Evolving midpoint, floor, ceiling
 'pq_func_var[2]': 'Mh',
 'pq_func_var2[2]': '1+z',
 'pq_val_ceil[2]': 1,
 'pq_val_floor[2]': 0,
 'pq_func_par0[2]': 0,
 'pq_func_par1[2]': 0.85,
 'pq_func_par2[2]': 12.2,
 'pq_func_par3[2]': -0.7,
 'pq_func_par4[2]': 0.5,    # terms that scale (1 - a)
 'pq_func_par5[2]': 0.1,    # terms that scale (1 - a)
 'pq_func_par6[2]': 0.3,    # terms that scale (1 - a)
 'pq_func_par7[2]': 0,      # terms that scale (1 - a)
 'pq_func_par8[2]': 0,      # terms that scale log(1+z)
 'pq_func_par9[2]': 0,      # terms that scale log(1+z)
 'pq_func_par10[2]': 0,     # terms that scale log(1+z)
 'pq_func_par11[2]': 0,     # terms that scale log(1+z)
 'pq_func_par12[2]': 0,     # terms that scale z
 'pq_func_par13[2]': 0,     # terms that scale z
 'pq_func_par14[2]': 0,     # terms that scale z
 'pq_func_par15[2]': 0,     # terms that scale z
}

#centrals_sf.update(_base)

centrals_q = centrals_sf.copy()
centrals_q['pop_sfh'] = 'ssp'
centrals_q['pop_aging'] = True
centrals_q['pop_ssfr'] = None
centrals_q['pop_ssp'] = True
centrals_q['pop_age'] = 1e4
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

# This results in a Z14-like amount of IHL
subhalos['pop_fsurv{2}'] = 'pq[3]'
subhalos['pop_fsurv_inv{2}'] = False
subhalos['pq_func[3]{2}'] = 'erf_evolB13'
subhalos['pq_func_var[3]{2}'] = 'Mh'
subhalos['pq_func_var2[3]{2}'] = '1+z'
subhalos['pq_val_ceil[3]{2}'] = 1
subhalos['pq_val_floor[3]{2}'] = 0
subhalos['pq_func_par0[3]{2}'] = 0.0  # step = par0-par1
subhalos['pq_func_par1[3]{2}'] = 1    # fsurv = par1 + step * tanh(stuff)
subhalos['pq_func_par2[3]{2}'] = 11.5
subhalos['pq_func_par3[3]{2}'] = 1 # dlogM
subhalos['pq_func_par4[3]{2}'] = 1.  # Pin to z=0
subhalos['pq_func_par5[3]{2}'] = 0
subhalos['pq_func_par6[3]{2}'] = 0
subhalos['pq_func_par7[3]{2}'] = 0
subhalos['pq_func_par8[3]{2}'] = 0
subhalos['pq_func_par9[3]{2}'] = 0
subhalos['pq_func_par10[3]{2}'] = 0
subhalos['pq_func_par11[3]{2}'] = 0
subhalos['pq_func_par12[3]{2}'] = 0
subhalos['pq_func_par13[3]{2}'] = 0
subhalos['pq_func_par14[3]{2}'] = 0
subhalos['pq_func_par15[3]{2}'] = 0
subhalos['pq_func_par16[3]{2}'] = 0

subhalos['pop_fsurv{3}'] = 'link:fsurv:2'
subhalos['pop_fsurv_inv{3}'] = True

subhalo_focc = {
 'pop_focc': 'pq[20]',
 'pq_func[20]': 'erf',
 'pq_func_var[20]': 'Mh',
 'pq_func_par0[20]': 0,
 'pq_func_par1[20]': 1,
 'pq_func_par2[20]': 8,
 'pq_func_par3[20]': 1,
}

maximal_ihl = {'pop_focc{2}': 1, 'pop_fsurv{2}': 0}
minimal_ihl = {'pop_focc{2}': 1, 'pop_fsurv{2}': 1}

ihl_like_z14 = {}
ihl_like_z14['pq_func_par0[3]{2}'] = 0.00 # step = par0-par1
ihl_like_z14['pq_func_par1[3]{2}'] = 1    # fsurv = par1 + step * tanh(stuff)
ihl_like_z14['pq_func_par2[3]{2}'] = 11.3
ihl_like_z14['pq_func_par3[3]{2}'] = 0.8 # dlogM
ihl_like_z14['pq_func_par4[3]{2}'] = 1.  # Pin to z=0
ihl_like_z14['pq_func_par5[3]{2}'] = -1. # Evolves as (1+z)^{-1}

dust = {}
dust['pop_dust_template'] = 'WD01:MWRV31'
dust['pop_Av'] = 'pq[4]'
dust['pq_func[4]'] = 'pl_evolN'
dust['pq_func_var[4]'] = 'Ms'
dust['pq_func_var2[4]'] = '1+z'
dust['pq_func_par0[4]'] = 0.   # Off by default
dust['pq_func_par1[4]'] = 1e10
dust['pq_func_par2[4]'] = 0.2
dust['pq_func_par3[4]'] = 1.     # Anchored to z=0
dust['pq_func_par4[4]'] = 0   # no evolution yet.
dust['pq_val_floor[4]'] = 0

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
 'pop_Z': 0.02,#('mzr', 0.02),
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

smhm_Q = {}
smhm_Q['pop_fstar{1}'] = 'pq[10]'
smhm_Q['pq_func[10]{1}'] = 'dpl_evolB13'
smhm_Q['pq_func_var[10]{1}'] = 'Mh'
smhm_Q['pq_func_var2[10]{1}'] = '1+z'
smhm_Q['pq_func_par0[10]{1}'] = 9.7957e-04
smhm_Q['pq_func_par1[10]{1}'] = 8.7620e+11
smhm_Q['pq_func_par2[10]{1}'] = 8.1798e-01
smhm_Q['pq_func_par3[10]{1}'] = -7.2136e-01
smhm_Q['pq_func_par4[10]{1}'] = 1e10
smhm_Q['pq_func_par5[10]{1}'] = 1.     # pivot in 1+z
smhm_Q['pq_func_par6[10]{1}'] = -1.7136e-01
smhm_Q['pq_func_par7[10]{1}'] = 1.1776e-01
smhm_Q['pq_func_par8[10]{1}'] = 5.3506e-01
smhm_Q['pq_func_par9[10]{1}'] = -9.1944e-01
smhm_Q['pq_func_par10[10]{1}'] = 0.0
smhm_Q['pq_func_par11[10]{1}'] = 0.0
smhm_Q['pq_func_par12[10]{1}'] = 0.0
smhm_Q['pq_func_par13[10]{1}'] = 0.0
smhm_Q['pq_func_par14[10]{1}'] = 0.0
smhm_Q['pq_func_par15[10]{1}'] = 0.0
smhm_Q['pq_func_par16[10]{1}'] = 0.0
smhm_Q['pq_val_ceil[10]{1}'] = 1

smhm_best = \
{
'pq_func_par0[0]{0}': 2.1813e-03,
'pq_func_par1[0]{0}': 7.5109e+11,
'pq_func_par2[0]{0}': 9.1506e-01,
'pq_func_par3[0]{0}': -6.4217e-02,
'pq_func_par0[10]{1}': 1.6234e-05,
'pq_func_par1[10]{1}': 1.2998e+12,
'pq_func_par2[10]{1}': 1.5670e+00,
'pq_func_par3[10]{1}': -5.9479e-01,
'pq_func_par5[0]{0}': -8.3219e-01,
'pq_func_par6[0]{0}': -1.4254e-01,
'pq_func_par7[0]{0}': 4.9219e-01,
'pq_func_par8[0]{0}': 1.5737e-02,
'pq_func_par9[0]{0}': -1.6946e-01,
'pq_func_par10[0]{0}': 5.9837e-03,
'pq_func_par11[0]{0}': -5.5587e-02,
'pq_func_par12[0]{0}': -5.6498e-01,
'pq_func_par13[0]{0}': 1.6355e-02,
'pq_func_par14[0]{0}': 3.4905e-01,
'pq_func_par15[0]{0}': -8.0572e-03,
'pq_func_par16[0]{0}': -1.2348e-02,
'pq_func_par5[10]{1}': 5.6279e-02,
'pq_func_par6[10]{1}': 2.6325e-01,
'pq_func_par7[10]{1}': -1.0397e-01,
'pq_func_par8[10]{1}': 5.3918e-01,
'pq_func_par9[10]{1}': -2.2759e-01,
'pq_func_par10[10]{1}': -5.8024e-01,
'pq_func_par11[10]{1}': -5.6683e-03,
'pq_func_par12[10]{1}': 1.3668e-02,
'pq_func_par13[10]{1}': 2.0778e-01,
'pq_func_par14[10]{1}': 1.2173e-01,
'pq_func_par15[10]{1}': -8.3519e-03,
'pq_func_par16[10]{1}': 7.0691e-04,
'pq_func_par0[2]{0}': 0.10473256877648432,
'pq_func_par1[2]{0}': 0.5226271922347238,
'pq_func_par2[2]{0}': 12.071847228837038,
'pq_func_par3[2]{0}': -0.10682927772213446,
'pq_func_par4[2]{0}': -0.7028407877245324,
'pq_func_par5[2]{0}': 0.7224105360651556,
'pq_func_par6[2]{0}': 0.6366107473353979,
'pq_func_par7[2]{0}': -0.5371916374513108,
'pq_func_par8[2]{0}': 0.0013330403659420594,
'pq_func_par9[2]{0}': -0.0015344268154752986,
'pq_func_par10[2]{0}': 0.0026575013451226675,
'pq_func_par11[2]{0}': 0.0029022726658339895,
'pq_func_par12[2]{0}': -0.0019447635180265543,
'pq_func_par13[2]{0}': 0.0031967605417705294,
'pq_func_par14[2]{0}': -0.004661816245667721,
'pq_func_par15[2]{0}': 0.004384020020400418
}
