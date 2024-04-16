import numpy as np
from ares.physics.Constants import E_LyA

_base = \
{
 "halo_dt": 100,
 "halo_tmin": 100.,
 "halo_tmax": 13.7e3, # Myr
 'halo_mf': 'Tinker10',
 "halo_mf_sub": 'Tinker08',

 # NIRB
 'tau_approx': 0,#'neutral',
 'tau_clumpy': 2,

 'cosmology_id': 'best',
 'cosmology_name': 'planck_TTTEEE_lowl_lowE',
 'cosmological_Mmin': None,

 'first_light_redshift': 10,
 'final_redshift': 1e-3,

 'tau_redshift_bins': 100,

 'halo_dlnk': 0.05,
 'halo_lnk_min': -9.,
 'halo_lnk_max': 11.,
}

centrals_sf = \
{
 'pop_use_lum_cache': True,
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
 'pq_func_par4[2]': 0,      # terms that scale (1 - a)
 'pq_func_par5[2]': 0,      # terms that scale (1 - a)
 'pq_func_par6[2]': 0,      # terms that scale (1 - a)
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
 'pq_func_par23[2]': 0.,   # evolution in Mc (par20)
 'pq_func_par24[2]': 0.,   # evolution in Mc (par20)
}

_ssfr_dpl = \
{
# sSFR(z, Mstell)
 'pop_ssfr': 'pq[1]',
 'pq_func[1]': 'dplx_evolB13',
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

for par in centrals_sf:
    if ('[0]' in par) or ('[1]' in par) or ('[2]' in par):
        del centrals_q[par]

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
satellites_sf['pop_focc'] = 'link:focc:0'
satellites_sf['pop_focc_inv'] = False
satellites_sf['pop_centrals'] = 0
satellites_sf['pop_centrals_id'] = 0
satellites_sf['pop_prof_1h'] = 'nfw'
satellites_sf['pop_include_1h'] = True
satellites_sf['pop_include_2h'] = True
satellites_sf['pop_include_shot'] = True
satellites_sf['pop_fstar'] = 'link:fstar:0'
for par in centrals_sf:
    if ('[0]' in par)  or ('[1]' in par) or ('[2]' in par):
        del satellites_sf[par]

satellites_sf['pop_sfr'] = 'link:sfr:0'

satellites_q = centrals_q.copy()
satellites_q['pop_focc'] = 'link:focc:2'
satellites_q['pop_focc_inv'] = True
satellites_q['pop_centrals'] = 0
satellites_q['pop_centrals_id'] = 0
satellites_q['pop_prof_1h'] = 'nfw'
satellites_q['pop_include_1h'] = True
satellites_q['pop_include_2h'] = True
satellites_q['pop_include_shot'] = True
satellites_q['pop_fstar'] = 'link:fstar:0'
satellites_q['pop_ssfr'] = None

satellites_q['pop_sfh'] = 'ssp'
satellites_q['pop_aging'] = True
satellites_q['pop_ssp'] = True
satellites_q['pop_age'] = 5e3
satellites_q['pop_Z'] = 0.02


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

# Dust
dust = {}
dust['pop_dust_template'] = 'C00'
dust['pop_Av'] = 'pq[4]'
dust['pq_func[4]'] = 'pl_evolB13'
dust['pq_func_var[4]'] = 'Ms'
dust['pq_func_var2[4]'] = '1+z'
dust['pq_func_par0[4]'] = 0    # Off by default
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

for par in dust.keys():
    base[par + '{0}'] = dust[par]

base_centrals = base.copy()

# This results in a Z14-like amount of IHL
subhalos['pop_fsurv{2}'] = 1#
subhalos_focc = {}
subhalos_focc['pop_fsurv{2}'] = 'pq[3]'
subhalos_focc['pop_fsurv_inv{2}'] = False
subhalos_focc['pq_func[3]{2}'] = 'erf_evolB13'
subhalos_focc['pq_func_var[3]{2}'] = 'Mh'
subhalos_focc['pq_func_var2[3]{2}'] = '1+z'
subhalos_focc['pq_val_ceil[3]{2}'] = 1
subhalos_focc['pq_val_floor[3]{2}'] = 0
subhalos_focc['pq_func_par0[3]{2}'] = 0.0  # step = par0-par1
subhalos_focc['pq_func_par1[3]{2}'] = 1    # fsurv = par1 + step * tanh(stuff)
subhalos_focc['pq_func_par2[3]{2}'] = 11.5
subhalos_focc['pq_func_par3[3]{2}'] = 1 # dlogM
subhalos_focc['pq_func_par4[3]{2}'] = 1.  # Pin to z=0
subhalos_focc['pq_func_par5[3]{2}'] = 0
subhalos_focc['pq_func_par6[3]{2}'] = 0
subhalos_focc['pq_func_par7[3]{2}'] = 0
subhalos_focc['pq_func_par8[3]{2}'] = 0
subhalos_focc['pq_func_par9[3]{2}'] = 0
subhalos_focc['pq_func_par10[3]{2}'] = 0
subhalos_focc['pq_func_par11[3]{2}'] = 0
subhalos_focc['pq_func_par12[3]{2}'] = 0
subhalos_focc['pq_func_par13[3]{2}'] = 0
subhalos_focc['pq_func_par14[3]{2}'] = 0
subhalos_focc['pq_func_par15[3]{2}'] = 0
subhalos_focc['pq_func_par16[3]{2}'] = 0
subhalos_focc['pq_func_par17[3]{2}'] = 0
subhalos_focc['pq_func_par18[3]{2}'] = 0
subhalos_focc['pq_func_par19[3]{2}'] = 0
subhalos_focc['pq_func_par20[3]{2}'] = 0

# Dust
subhalos['pop_Av{2}'] = 'link:Av:0'
subhalos['pop_dust_template{2}'] = 'WD01:MWRV31'

subhalos['pop_fsurv{3}'] = 'link:fsurv:2'
subhalos['pop_fsurv_inv{3}'] = False

ihl = {}
ihl['pop_sfr_model{4}'] = 'smhm-func'
ihl['pop_solve_rte{4}'] = (0.12, E_LyA)
ihl['pop_Emin{4}'] = 0.12
ihl['pop_Emax{4}'] = 24.6
ihl['pop_zdead{4}'] = 0

# SED info
ihl['pop_sed{4}'] = 'eldridge2017'
ihl['pop_rad_yield{4}'] = 'from_sed'

ihl['pop_sed_degrade{4}'] = 10
ihl['pop_nebular{4}'] = 0

#
ihl['pop_centrals{4}'] = False
ihl['pop_centrals_id{4}'] = 0
ihl['pop_fstar{4}'] = 'link:fstar:2'
ihl['pop_focc{4}'] = 1
ihl['pop_fsurv{4}'] = 'link:fsurv:2'
ihl['pop_fsurv_inv{4}'] = True
ihl['pop_include_1h{4}'] = True
ihl['pop_include_2h{4}'] = True
ihl['pop_include_shot{4}'] = False
ihl['pop_Mmin{4}'] = 1e10
ihl['pop_Tmin{4}'] = None
ihl['pop_sfh{4}'] = 'ssp'
ihl['pop_aging{4}'] = True
ihl['pop_ssfr{4}'] = None
ihl['pop_sfr{4}'] = None
ihl['pop_ssp{4}'] = True
ihl['pop_age{4}'] = 1e4
ihl['pop_Z{4}'] = 0.02

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

best = \
{
'pq_func_par0[0]{0}': 3.8954e-04,
'pq_func_par1[0]{0}': 5.3712e+11,
'pq_func_par2[0]{0}': 1.1115e+00,
'pq_func_par3[0]{0}': -4.3636e-01,
'pq_func_par5[0]{0}': 3.8174e-02,
'pq_func_par9[0]{0}': 1.1444e-01,
'pq_func_par13[0]{0}': -3.5468e-01,
'pq_func_par6[0]{0}': 2.2669e-01,
'pq_func_par10[0]{0}': 3.3021e-01,
'pq_func_par14[0]{0}': 2.0931e-02,
'pq_func_par0[2]{0}': 3.6690e-02,
'pq_func_par1[2]{0}': 7.7328e-01,
'pq_func_par2[2]{0}': 1.2209e+01,
'pq_func_par3[2]{0}': -6.6942e-01,
'pq_func_par4[2]{0}': 6.1058e-01,
'pq_func_par8[2]{0}': 4.2276e-02,
'pq_func_par12[2]{0}': -9.1205e-04,
'pq_func_par5[2]{0}': 2.9010e-01,
'pq_func_par9[2]{0}': -2.4559e-02,
'pq_func_par13[2]{0}': 3.5421e-02,
'pq_func_par6[2]{0}': -2.8269e-01,
'pq_func_par10[2]{0}': 7.6175e-02,
'pq_func_par14[2]{0}': -2.1064e-02,
'pq_func_par7[2]{0}': 8.5306e-01,
'pq_func_par11[2]{0}': -1.7278e-02,
'pq_func_par15[2]{0}': -4.9272e-02,
'pq_func_par0[1]{0}': 1.5927e-02,
'pq_func_par1[1]{0}': 7.1008e+10,
'pq_func_par2[1]{0}': 1.5150e+00,
'pq_func_par3[1]{0}': 4.5578e-01,
'pq_func_par5[1]{0}': 2.0392e-02,
'pq_func_par9[1]{0}': 2.1654e-01,
'pq_func_par13[1]{0}': 7.9087e-02,
'pq_func_par6[1]{0}': 2.5633e+00,
'pq_func_par10[1]{0}': -9.8255e-02,
'pq_func_par14[1]{0}': -2.0452e-01,
'pq_func_par0[4]{0}': 1.1732e+00,
'pq_func_par2[4]{0}': 1.3039e-02,
'pq_func_par3[4]{0}': -9.3737e-03,
'pq_func_par5[4]{0}': -2.6789e-01,
'pq_func_par7[4]{0}': -7.4356e-02,
'pq_func_par4[4]{0}': 7.1479e-03,
'pq_func_par6[4]{0}': 3.5031e-02,
'pq_func_par8[4]{0}': 1.3915e-02,
'pq_func_par0[3]{2}': 3.0129e-01,
'pq_func_par1[3]{2}': 7.8550e-01,
'pq_func_par2[3]{2}': 1.2816e+01,
'pq_func_par3[3]{2}': 3.4158e+00,
'pq_func_par4[3]{2}': -1.9960e+00,
'pq_func_par5[3]{2}': 1.1755e+00,
'pq_func_par6[3]{2}': -6.3000e-01,
'pq_func_par7[3]{2}': 9.7541e-01,
'mu_0': 4.4865e-01,
'mu_a': -2.4907e-01,
'mu_z': -4.3745e-01,
'kappa_0': -5.6027e-02,
'kappa_a': 3.3710e-01,
'kappa_z': -2.6643e-01,
}

base.update(subhalos)
#base.update(ihl)
base.update(best)

##
# Allows subhalos to have different SMHM than centrals
subhalos_smhm_ext = {}
subhalos_smhm_ext['pop_fstar{2}'] = 'pq[5]'
subhalos_smhm_ext['pq_func[5]{2}'] = 'dplx_evolB13'
subhalos_smhm_ext['pq_func_var[5]{2}'] = 'Mh'
subhalos_smhm_ext['pq_func_var2[5]{2}'] = '1+z'

for i in range(0, 27):
    subhalos_smhm_ext['pq_func_par%i[5]{2}' % i] = base['pq_func_par%i[0]{0}' % i]

subhalos_smhm_ext['pop_fstar{3}'] = 'link:fstar:2'

##
# Allows subhalos to have different SFR than centrals
subhalos_sfr_ext = {}
subhalos_sfr_ext['pop_sfr{2}'] = 'pq[6]'
subhalos_sfr_ext['pq_func[6]{2}'] = 'dplx_evolB13'
subhalos_sfr_ext['pq_func_var[6]{2}'] = 'Mh'
subhalos_sfr_ext['pq_func_var2[6]{2}'] = '1+z'


for i in range(0, 27):
    subhalos_sfr_ext['pq_func_par%i[6]{2}' % i] = base['pq_func_par%i[1]{0}' % i]

##
# Allows subhalos to have different quenched fraction than centrals
subhalos_focc_ext = {}
subhalos_focc_ext['pop_focc{2}'] = 'pq[7]'
subhalos_focc_ext['pq_func[7]{2}'] = 'erf_evolB13'
subhalos_focc_ext['pq_val_ceil[7]{2}'] = 1
subhalos_focc_ext['pq_func_var[7]{2}'] = 'Mh'
subhalos_focc_ext['pq_func_var2[7]{2}'] = '1+z'
subhalos_focc_ext['pq_func_par0[7]{2}'] = 0.
subhalos_focc_ext['pq_func_par1[7]{2}'] = 0.85
subhalos_focc_ext['pq_func_par2[7]{2}'] = 12.2
subhalos_focc_ext['pq_func_par3[7]{2}'] = -0.7
for i in range(4, 26):
    subhalos_focc_ext['pq_func_par%i[7]{2}' % i] = 0

subhalos_focc_ext['pop_focc{3}'] = 'link:focc:2'
subhalos_focc_ext['pop_focc_inv{3}'] = True

sed_modeling = {}
sed_modeling['pop_age{0}'] = 1e2, 'hubble'
sed_modeling['pop_age{2}'] = 1e2, 'hubble'
sed_modeling['pop_age{1}'] = 'hubble'
sed_modeling['pop_age{3}'] = 'hubble'

sed_modeling['pop_sfh{2}'] = 'constant+ssp'
sed_modeling['pop_ssp{2}'] = (False, True)

sed_modeling['pop_age_definition{0}'] = 'mixed'
sed_modeling['pop_age_definition{1}'] = 'mixed'
sed_modeling['pop_age_definition{2}'] = 'mixed'
sed_modeling['pop_age_definition{3}'] = 'mixed'

sed_modeling['pop_lum_corr{1}'] = 'sed_corrections_qgs_below_100_obs.hdf5'
sed_modeling['pop_lum_corr{3}'] = 'sed_corrections_qgs_below_100_obs.hdf5'
sed_modeling['pop_Z{1}'] = 0.02
sed_modeling['pop_Z{3}'] = 0.02

_mzr02 = {}
for par in mzr:
    _mzr02[par + '{0}'] = mzr[par]
    _mzr02[par + '{2}'] = mzr[par]

sed_modeling.update(_mzr02)

sed_modeling['pop_lum_corr{0}'] = 'sed_corrections_sfgs_mzr_obs.hdf5'
sed_modeling['pop_lum_corr{2}'] = 'sed_corrections_sfgs_mzr_obs.hdf5'

# Scaling relationships for common strong lines
# Each pair is rest wavelength [Angstroms] and L_line [erg/s/(Msun/yr)]
lines = {}
# (1216, 6.85e41),             # Ly-a
lines['pop_lum_per_sfr_at_wave{0}'] = \
    [
     (6563, 1.27e41),             # H-alpha
     (5007, 1.32e41),             # [O III]
     (4861, 0.44e41),             # H-beta
     (4340, 0.468 * 0.44e41),     # H-gamma
     (4102, 0.259 * 0.44e41),     # H-delta
     (3970, 0.159 * 0.44e41),     # H-epsilon
     (3727, 0.71e41),             # [O II]
     (1.87e4, 1.27e41 * 0.123)]   # [P-alpha]
lines['pop_lum_per_sfr_at_wave{2}'] = lines['pop_lum_per_sfr_at_wave{0}']
