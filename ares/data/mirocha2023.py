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
 'tau_clumpy': 0,

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
 'pop_sed_degrade': 10,

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
 'pq_func[0]': 'dplx_evolB13',
 'pq_func_var[0]': 'Mh',
 'pq_func_var2[0]': '1+z',
 'pq_func_par0[0]': 0.0003,
 'pq_func_par1[0]': 1.5e12,
 'pq_func_par2[0]': 1,
 'pq_func_par3[0]': -0.6,
 'pq_func_par4[0]': 1e10,           # normalization pinned to this Mh
 'pq_func_par5[0]': 0,              # norm
 'pq_func_par6[0]': 0,              # peak
 'pq_func_par7[0]': 0,              # low
 'pq_func_par8[0]': 0,              # high
 'pq_func_par9[0]': 0.0,            # norm
 'pq_func_par10[0]': 0.0,           # peak
 'pq_func_par11[0]': 0.0,           # low
 'pq_func_par12[0]': 0.0,           # high
 'pq_func_par13[0]': 0.0,           # norm
 'pq_func_par14[0]': 0.0,           # peak
 'pq_func_par15[0]': 0.0,           # low
 'pq_func_par16[0]': 0.0,           # high
 'pq_func_par17[0]': 0.0,           # norm
 'pq_func_par18[0]': 0.0,           # peak
 'pq_func_par19[0]': 0.0,           # low
 'pq_func_par20[0]': 0.0,           # high

 # Extension!
 'pq_func_par21[0]': 5.0, # evolution done in log10(Mturn), hence default > 0
 'pq_func_par22[0]': 0.0,
 'pq_func_par23[0]': 0.0,
 'pq_func_par24[0]': 0.0,
 'pq_func_par25[0]': 0.0,
 'pq_func_par26[0]': 0.0,

 'pq_val_ceil[0]': 1,

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
 'pq_func_par16[2]': 0,     # terms that scale a
 'pq_func_par17[2]': 0,     # terms that scale a
 'pq_func_par18[2]': 0,     # terms that scale a
 'pq_func_par19[2]': 0,     # terms that scale a
}

focc_erfx = \
{
 'pq_func_par20[2]': 1e11,
 'pq_func_par21[2]': -0.1,
 'pq_func_par22[2]': 0.1,
}

_ssfr_dpl = \
{
# sSFR(z, Mstell)
 'pop_ssfr': 'pq[1]',
 'pq_func[1]': 'dpl_evolB13',
 'pq_func_var[1]': 'Ms',
 'pq_func_var2[1]': '1+z',
 'pq_func_par0[1]': 5e-10,
 'pq_func_par1[1]': 1e5,
 'pq_func_par2[1]': 0,
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
 'pq_func_par17[1]': 0.0,
 'pq_func_par18[1]': 0.0,
 'pq_func_par19[1]': 0.0,
 'pq_func_par20[1]': 0.0,
}

_sfr_dpl = \
{
# sSFR(z, Mstell)
 'pop_sfr': 'pq[1]',
 'pq_func[1]': 'dplx_evolB13',
 'pq_func_var[1]': 'Mh',
 'pq_func_var2[1]': '1+z',
 'pq_func_par0[1]': 0.01,
 'pq_func_par1[1]': 3e12,
 'pq_func_par2[1]': 1.6,
 'pq_func_par3[1]': 0.2,
 'pq_func_par4[1]': 1e10,  # Mh anchor
 'pq_func_par5[1]': 0.6,    # scales (1-a) term
 'pq_func_par6[1]': 0.,     # scales (1-a) term
 'pq_func_par7[1]': 0,      # scales (1-a) term
 'pq_func_par8[1]': 0,      # scales (1-a) term
 'pq_func_par9[1]': 0.,     # scales log(1+z) term
 'pq_func_par10[1]': 0.0,   # scales log(1+z) term
 'pq_func_par11[1]': 0.0,   # scales log(1+z) term
 'pq_func_par12[1]': 0.0,   # scales log(1+z) term
 'pq_func_par13[1]': 0.0,
 'pq_func_par14[1]': 0.0,
 'pq_func_par15[1]': 0.0,
 'pq_func_par16[1]': 0.0,
 'pq_func_par17[1]': 0.0,
 'pq_func_par18[1]': 0.0,
 'pq_func_par19[1]': 0.0,
 'pq_func_par20[1]': 0.0,
 # Extension!
 'pq_func_par21[1]': 0.0,
 'pq_func_par22[1]': 0.0,
 'pq_func_par23[1]': 0.0,
 'pq_func_par24[1]': 0.0,
 'pq_func_par25[1]': 0.0,
 'pq_func_par26[1]': 0.0,
}

centrals_sf.update(_sfr_dpl)

centrals_q = centrals_sf.copy()
centrals_q['pop_sfh'] = 'ssp'
centrals_q['pop_aging'] = True
centrals_q['pop_ssfr'] = None
centrals_q['pop_sfr'] = None
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
satellites_sf['pop_include_shot'] = True
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
subhalos['pq_func_par17[3]{2}'] = 0
subhalos['pq_func_par18[3]{2}'] = 0
subhalos['pq_func_par19[3]{2}'] = 0
subhalos['pq_func_par20[3]{2}'] = 0

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
ihl_like_z14['pq_func_par2[3]{2}'] = 9.3
ihl_like_z14['pq_func_par3[3]{2}'] = 0.8 # dlogM
ihl_like_z14['pq_func_par4[3]{2}'] = 1.  # Pin to z=0
ihl_like_z14['pq_func_par5[3]{2}'] = -1. # Evolves as (1+z)^{-1}

dust = {}
dust['pop_dust_template'] = 'WD01:MWRV31'
dust['pop_Av'] = 'pq[4]'
dust['pq_func[4]'] = 'pl_evolB13'
dust['pq_func_var[4]'] = 'Ms'
dust['pq_func_var2[4]'] = '1+z'
dust['pq_func_par0[4]'] = 0.   # Off by default
dust['pq_func_par1[4]'] = 1e10
dust['pq_func_par2[4]'] = 0.2
dust['pq_func_par3[4]'] = 0   # no evolution yet.
dust['pq_func_par4[4]'] = 0   # no evolution yet.
dust['pq_func_par5[4]'] = 0   # no evolution yet.
dust['pq_func_par6[4]'] = 0   # no evolution yet.
dust['pq_func_par7[4]'] = 0   # no evolution yet.
dust['pq_func_par8[4]'] = 0   # no evolution yet.
dust['pq_func_par9[4]'] = 0   # no evolution yet.
dust['pq_func_par10[4]'] = 0   # no evolution yet.
dust['pq_val_floor[4]'] = 0

mzr = \
{
 'pop_enrichment': True,
 'pop_mzr': 'pq[30]',
 'pop_fox': 0.03,
 "pq_func[30]": 'linear_evolN',
 'pq_func_var[30]': 'Ms',
 'pq_func_var2[30]': '1+z',
 'pq_func_par0[30]': 8.75,
 'pq_func_par1[30]': 10,
 'pq_func_par2[30]': 0.25,
 'pq_func_par3[30]': 1.,     # pin to z=0
 'pq_func_par4[30]': -0.1,   # mild evolution
 'pq_val_ceil[30]': 8.8,
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
smhm_Q['pq_func_par5[10]{1}'] = 0.
smhm_Q['pq_func_par6[10]{1}'] = 0.
smhm_Q['pq_func_par7[10]{1}'] = 0.
smhm_Q['pq_func_par8[10]{1}'] = 0.
smhm_Q['pq_func_par9[10]{1}'] = 0.
smhm_Q['pq_func_par10[10]{1}'] = 0.0
smhm_Q['pq_func_par11[10]{1}'] = 0.0
smhm_Q['pq_func_par12[10]{1}'] = 0.0
smhm_Q['pq_func_par13[10]{1}'] = 0.0
smhm_Q['pq_func_par14[10]{1}'] = 0.0
smhm_Q['pq_func_par15[10]{1}'] = 0.0
smhm_Q['pq_func_par16[10]{1}'] = 0.0
smhm_Q['pq_func_par17[10]{1}'] = 0.0
smhm_Q['pq_func_par18[10]{1}'] = 0.0
smhm_Q['pq_func_par19[10]{1}'] = 0.0
smhm_Q['pq_func_par20[10]{1}'] = 0.0
smhm_Q['pq_val_ceil[10]{1}'] = 1

best_smfs_same = \
{
'pq_func_par0[0]{0}': 1.0281e-03,
'pq_func_par1[0]{0}': 5.2562e+11,
'pq_func_par2[0]{0}': 1.0288e+00,
'pq_func_par3[0]{0}': -5.9869e-01,
'pq_func_par5[0]{0}': 2.0982e+00,
'pq_func_par6[0]{0}': -1.5912e+00,
'pq_func_par9[0]{0}': -3.4821e+00,
'pq_func_par10[0]{0}': 2.3568e+00,
'pq_func_par13[0]{0}': 8.0858e-01,
'pq_func_par14[0]{0}': -4.1392e-01,

'pq_func_par0[2]{0}': 9.4945e-02,
'pq_func_par1[2]{0}': 9.5740e-01,
'pq_func_par2[2]{0}': 1.1919e+01,
'pq_func_par3[2]{0}': -5.1173e-01,
'pq_func_par4[2]{0}': 6.4382e-01,
'pq_func_par5[2]{0}': -2.1909e+00,
'pq_func_par6[2]{0}': 9.9856e-01,
'pq_func_par7[2]{0}': 3.9622e-01,
'pq_func_par8[2]{0}': -2.4369e-02,
'pq_func_par9[2]{0}': 1.5689e+00,
'pq_func_par10[2]{0}': -8.9817e-01,
'pq_func_par11[2]{0}': 1.9932e-01,
'pq_func_par12[2]{0}': 2.0178e-02,
'pq_func_par13[2]{0}': -9.6158e-03,
'pq_func_par14[2]{0}': 2.1655e-01,
'pq_func_par15[2]{0}': -1.0448e-01,
}

best_smfs_diff = \
{
'pq_func_par0[0]{0}': 9.3949e-04,
'pq_func_par1[0]{0}': 1.0632e+12,
'pq_func_par2[0]{0}': 1.0366e+00,
'pq_func_par3[0]{0}': -7.2380e-01,
'pq_func_par0[10]{1}': 5.2510e-04,
'pq_func_par1[10]{1}': 5.7532e+11,
'pq_func_par2[10]{1}': 9.8410e-01,
'pq_func_par3[10]{1}': -1.4856e-01,
'pq_func_par5[0]{0}': -6.3509e-01,
'pq_func_par9[0]{0}': -2.3475e-02,
'pq_func_par13[0]{0}': -1.5872e-01,
'pq_func_par6[0]{0}': -1.0569e-01,
'pq_func_par10[0]{0}': 9.2388e-02,
'pq_func_par14[0]{0}': 2.8008e-01,
'pq_func_par5[10]{1}': -1.1792e-01,
'pq_func_par9[10]{1}': 1.4956e-01,
'pq_func_par13[10]{1}': -1.2040e-01,
'pq_func_par6[10]{1}': 3.3779e-02,
'pq_func_par10[10]{1}': 2.8677e-01,
'pq_func_par14[10]{1}': -1.8024e-01,
'pq_func_par0[2]{0}': 8.2985e-01,
'pq_func_par1[2]{0}': 6.7249e-01,
'pq_func_par2[2]{0}': 1.1666e+01,
'pq_func_par3[2]{0}': -2.4143e-01,
'pq_func_par4[2]{0}': -3.0182e+00,
'pq_func_par8[2]{0}': 4.5449e-02,
'pq_func_par12[2]{0}': -2.1116e-03,
'pq_func_par5[2]{0}': 5.7532e-01,
'pq_func_par9[2]{0}': 1.5401e-02,
'pq_func_par13[2]{0}': -1.0251e-02,
'pq_func_par6[2]{0}': 1.2964e+00,
'pq_func_par10[2]{0}': -2.9521e-02,
'pq_func_par14[2]{0}': -9.9506e-03,
'pq_func_par7[2]{0}': -1.7596e-01,
'pq_func_par11[2]{0}': -1.7671e-02,
'pq_func_par15[2]{0}': -1.2388e-02,
}

best_ssfr = \
{
'pq_func_par0[1]{0}': 7.2640e-10,
'pq_func_par1[1]{0}': 2.2075e+08,
'pq_func_par2[1]{0}': -7.8741e-02,
'pq_func_par3[1]{0}': -7.1286e-01,
'pq_func_par5[1]{0}': 1.3126e+00,
'pq_func_par6[1]{0}': -1.7541e+00,
'pq_func_par9[1]{0}': -3.0984e-01,
'pq_func_par10[1]{0}': 3.8616e+00,
'pq_func_par13[1]{0}': 7.2800e-02,
'pq_func_par14[1]{0}': -1.8903e-01,
}

best_sfr = \
{
'pq_func_par0[1]{0}': 7.4748e-03,
'pq_func_par1[1]{0}': 9.3513e+10,
'pq_func_par2[1]{0}': 1.8852e+00,
'pq_func_par3[1]{0}': 2.3410e-01,
'pq_func_par5[1]{0}': -8.0080e-01,
'pq_func_par9[1]{0}': 6.6126e-01,
'pq_func_par13[1]{0}': -1.2146e-02,
'pq_func_par6[1]{0}': 2.7706e+00,
'pq_func_par10[1]{0}': -1.6599e+00,
'pq_func_par14[1]{0}': 4.8501e-01,
'pq_func_par7[1]{0}': -6.7612e+00,
'pq_func_par11[1]{0}': 7.6991e+00,
'pq_func_par15[1]{0}': -1.8067e+00,
'pq_func_par8[1]{0}': -4.3245e+00,
'pq_func_par12[1]{0}': 4.6987e+00,
'pq_func_par16[1]{0}': -1.0069e+00,
}

#base.update(best_smf_same)
#base.update(best_sfr)

best = \
{
'pq_func_par0[0]{0}': 1.4528e-03,
'pq_func_par1[0]{0}': 8.0368e+11,
'pq_func_par2[0]{0}': 1.0138e+00,
'pq_func_par3[0]{0}': -5.4139e-01,
'pq_func_par5[0]{0}': -4.4920e-01,
'pq_func_par9[0]{0}': -3.1000e-01,
'pq_func_par13[0]{0}': -1.5618e-01,
'pq_func_par6[0]{0}': -3.6642e-01,
'pq_func_par10[0]{0}': 3.9501e-01,
'pq_func_par14[0]{0}': 1.4752e-01,
'pq_func_par0[2]{0}': 6.8249e-02,
'pq_func_par1[2]{0}': 6.7548e-01,
'pq_func_par2[2]{0}': 1.1821e+01,
'pq_func_par3[2]{0}': -2.8344e-01,
'pq_func_par4[2]{0}': 8.3226e-01,
'pq_func_par8[2]{0}': 1.9903e-02,
'pq_func_par12[2]{0}': -6.6032e-02,
'pq_func_par5[2]{0}': 2.4409e-01,
'pq_func_par9[2]{0}': 1.3869e-01,
'pq_func_par13[2]{0}': 4.0331e-02,
'pq_func_par6[2]{0}': 4.3713e-01,
'pq_func_par10[2]{0}': 4.7108e-02,
'pq_func_par14[2]{0}': -4.6672e-02,
'pq_func_par7[2]{0}': -7.9427e-02,
'pq_func_par11[2]{0}': 3.8848e-02,
'pq_func_par15[2]{0}': 8.2503e-03,
'pq_func_par0[1]{0}': 1.0485e-02,
'pq_func_par1[1]{0}': 9.1208e+10,
'pq_func_par2[1]{0}': 1.8199e+00,
'pq_func_par3[1]{0}': 3.7529e-01,
'pq_func_par5[1]{0}': -2.2829e-01,
'pq_func_par9[1]{0}': 2.4969e-01,
'pq_func_par13[1]{0}': 5.7209e-02,
'pq_func_par6[1]{0}': 1.3389e+00,
'pq_func_par10[1]{0}': -1.0108e-01,
'pq_func_par14[1]{0}': -5.2890e-02,
'pq_func_par7[1]{0}': 7.3728e-02,
'pq_func_par11[1]{0}': -1.2489e-01,
'pq_func_par15[1]{0}': -2.4701e-02,
'pq_func_par8[1]{0}': 8.7978e-02,
'pq_func_par12[1]{0}': 5.6333e-02,
'pq_func_par16[1]{0}': 2.1005e-02,
'pq_func_par0[4]{0}': 9.2995e-01,
'pq_func_par2[4]{0}': 6.5168e-02,
'pq_func_par3[4]{0}': 8.7186e-01,
'pq_func_par5[4]{0}': -6.5286e-01,
'pq_func_par7[4]{0}': -5.0222e-02,
'pq_func_par4[4]{0}': -1.4911e+00,
'pq_func_par6[4]{0}': 1.2853e+00,
'pq_func_par8[4]{0}': -2.0233e-01,
'mu_0': -3.0337e-01,
'mu_a': 4.9473e-01,
'kappa': 1.2483e-01,
}

base.update(best)
