import os
import numpy as np
from ares.physics.Constants import E_LyA, lsun

HOME = os.getenv("HOME")

setup = \
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

 'first_light_redshift': 15,
 'final_redshift': 5e-3,

 'tau_redshift_bins': 100,

 'halo_dlnk': 0.05,
 'halo_lnk_min': -9.,
 'halo_lnk_max': 11.,
}

centrals_sf = \
{
 'pop_use_lum_cache': True,
 'pop_emissivity_tricks': False,
 'pop_sfr_model': 'smhm-func',
 'pop_solve_rte': (0.12, 13.6),
 'pop_Emin': 0.12,
 #'pop_Emax': E_LyA*0.999,
 #'pop_Emax': 24.6,
 'pop_Emax': 13.6,

 'pop_centrals': True,
 'pop_zdead': 0,
 'pop_include_1h': False,
 'pop_include_2h': True,
 'pop_include_shot': True,

 # SED info
 'pop_sed': 'bc03_2013',
 'pop_imf': 'chabrier',
 'pop_tracks': 'Padova1994',
 'pop_rad_yield': 'from_sed',

 'pop_fesc': 0.2,
 'pop_sed_degrade': None,

 'pop_nebular': 0,

 'pop_sfh': 'constant+ssp',
 'pop_ssp': (False, True),
 'pop_age': (100., 2.5e3),
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

 'pop_scatter_sfh': 0,

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

 # Systematics
 'pop_sys_method': 'separate',
 'pop_sys_mstell_now': 0,
 'pop_sys_mstell_a': 0,
 #'pop_sys_mstell_z': 0,
 'pop_sys_sfr_now': 0,
 'pop_sys_sfr_a': 0,
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
centrals_q['pop_scatter_sfh'] = 'pop_scatter_sfh{0}'
centrals_q['pop_sys_method'] = 'separate'
centrals_q['pop_sys_mstell_now'] = 'pop_sys_mstell_now{0}'
centrals_q['pop_sys_mstell_a'] = 'pop_sys_mstell_a{0}'
centrals_q['pop_sys_sfr_now'] = 'pop_sys_sfr_now{0}'
centrals_q['pop_sys_sfr_a'] = 'pop_sys_sfr_a{0}'

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
satellites_sf['pop_sys_mstell_now'] = 'pop_sys_mstell_now{0}'
satellites_sf['pop_sys_mstell_a'] = 'pop_sys_mstell_a{0}'
satellites_sf['pop_sys_sfr_now'] = 'pop_sys_sfr_now{0}'
satellites_sf['pop_sys_sfr_a'] = 'pop_sys_sfr_a{0}'


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
satellites_q['pop_scatter_sfh'] = 'pop_scatter_sfh{0}'

satellites_q['pop_sfh'] = 'ssp'
satellites_q['pop_aging'] = True
satellites_q['pop_ssp'] = True
satellites_q['pop_age'] = 1e4
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

_pop0 = centrals_sf.copy()
_pop1 = centrals_q.copy()
_pop2 = satellites_sf.copy()
_pop3 = satellites_q.copy()

for i, _pop in enumerate([_pop0, _pop1]):
    pf = {}
    for par in _pop.keys():
        pf[par + '{%i}' % i] = _pop[par]

    setup.update(pf)

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

dust_x = {}
dust_x['pop_dust_template_extension{0}'] = 'pq[40]'
dust_x['pq_func{0}[40]'] = 'pl_evolB13'
dust_x['pq_func_var{0}[40]'] = 'wave'
dust_x['pq_func_var2{0}[40]'] = '1+z'
dust_x['pq_func_par0{0}[40]'] = 1
dust_x['pq_func_par1{0}[40]'] = 5500
dust_x['pq_func_par2{0}[40]'] = 0.0
dust_x['pq_func_par3{0}[40]'] = 0     # norm
dust_x['pq_func_par4{0}[40]'] = 0     # slope
dust_x['pq_func_par5{0}[40]'] = 0     # norm
dust_x['pq_func_par6{0}[40]'] = 0     # slope
dust_x['pq_func_par7{0}[40]'] = 0     # norm
dust_x['pq_func_par8{0}[40]'] = 0     # slope
dust_x['pq_func_par9{0}[40]'] = 0     # slope
dust_x['pq_func_par10{0}[40]'] = 0     # slope

for par in dust.keys():
    setup[par + '{0}'] = dust[par]

dust_dpl = \
{
 'pq_func[4]{0}': 'dpl_evolB13',
 'pq_func_var[4]{0}': 'Ms',
 'pq_func_var2[4]{0}': '1+z',
 'pq_func_par0[4]{0}': 0.0,
 'pq_func_par1[4]{0}': 1e11,
 'pq_func_par2[4]{0}': 0.2,
 'pq_func_par3[4]{0}': 0.,
 'pq_func_par4[4]{0}': 1e10,           # normalization pinned to this Mh
 'pq_func_par5[4]{0}': 0,              # norm
 'pq_func_par6[4]{0}': 0,              # peak
 'pq_func_par7[4]{0}': 0,              # low
 'pq_func_par8[4]{0}': 0,              # high
 'pq_func_par9[4]{0}': 0.0,            # norm
 'pq_func_par10[4]{0}': 0.0,           # peak
 'pq_func_par11[4]{0}': 0.0,           # low
 'pq_func_par12[4]{0}': 0.0,           # high
 'pq_func_par13[4]{0}': 0.0,           # norm
 'pq_func_par14[4]{0}': 0.0,           # peak
 'pq_func_par15[4]{0}': 0.0,           # low
 'pq_func_par16[4]{0}': 0.0,           # high
 'pq_func_par17[4]{0}': 0.0,           # norm
 'pq_func_par18[4]{0}': 0.0,           # peak
 'pq_func_par19[4]{0}': 0.0,           # low
 'pq_func_par20[4]{0}': 0.0,           # high
}

base_centrals = setup.copy()

# This results in a Z14-like amount of IHL
subhalos['pop_fsurv{2}'] = 1#
subhalos_fsurv = {}
subhalos_fsurv['pop_fsurv{2}'] = 'pq[3]'
subhalos_fsurv['pop_fsurv_inv{2}'] = False
subhalos_fsurv['pq_func[3]{2}'] = 'erf_evolB13'
subhalos_fsurv['pq_func_var[3]{2}'] = 'Mh'
subhalos_fsurv['pq_func_var2[3]{2}'] = '1+z'
subhalos_fsurv['pq_val_ceil[3]{2}'] = 1
subhalos_fsurv['pq_val_floor[3]{2}'] = 0
subhalos_fsurv['pq_func_par0[3]{2}'] = 0.0  # step = par0-par1
subhalos_fsurv['pq_func_par1[3]{2}'] = 1    # fsurv = par1 + step * tanh(stuff)
subhalos_fsurv['pq_func_par2[3]{2}'] = 11.5
subhalos_fsurv['pq_func_par3[3]{2}'] = 1 # dlogM
subhalos_fsurv['pq_func_par4[3]{2}'] = 1.  # Pin to z=0
subhalos_fsurv['pq_func_par5[3]{2}'] = 0
subhalos_fsurv['pq_func_par6[3]{2}'] = 0
subhalos_fsurv['pq_func_par7[3]{2}'] = 0
subhalos_fsurv['pq_func_par8[3]{2}'] = 0
subhalos_fsurv['pq_func_par9[3]{2}'] = 0
subhalos_fsurv['pq_func_par10[3]{2}'] = 0
subhalos_fsurv['pq_func_par11[3]{2}'] = 0
subhalos_fsurv['pq_func_par12[3]{2}'] = 0
subhalos_fsurv['pq_func_par13[3]{2}'] = 0
subhalos_fsurv['pq_func_par14[3]{2}'] = 0
subhalos_fsurv['pq_func_par15[3]{2}'] = 0
subhalos_fsurv['pq_func_par16[3]{2}'] = 0
subhalos_fsurv['pq_func_par17[3]{2}'] = 0
subhalos_fsurv['pq_func_par18[3]{2}'] = 0
subhalos_fsurv['pq_func_par19[3]{2}'] = 0
subhalos_fsurv['pq_func_par20[3]{2}'] = 0

# Dust
subhalos['pop_Av{2}'] = 'link:Av:0'
subhalos['pop_dust_template{2}'] = 'C00'
#subhalos['pop_dust_template_extension{2}'] = 'C00'

subhalos['pop_fsurv{3}'] = 'link:fsurv:2'
subhalos['pop_fsurv_inv{3}'] = False

ihl = {}
ihl['pop_sfr_model{4}'] = 'smhm-func'
ihl['pop_solve_rte{4}'] = (0.12, E_LyA)
ihl['pop_Emin{4}'] = 0.12
ihl['pop_Emax{4}'] = 24.6
ihl['pop_zdead{4}'] = 0

# SED info
ihl['pop_sed{4}'] = 'bc03'
ihl['pop_rad_yield{4}'] = 'from_sed'

ihl['pop_sed_degrade{4}'] = None#10
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
ihl['pop_scatter_sfh{4}'] = 'pop_scatter_sfh{0}'

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

setup.update(subhalos)

##
# Allows subhalos to have different SMHM than centrals
subhalos_smhm_ext = {}
subhalos_smhm_ext['pop_fstar{2}'] = 'pq[5]'
subhalos_smhm_ext['pq_func[5]{2}'] = 'dplx_evolB13'
subhalos_smhm_ext['pq_func_var[5]{2}'] = 'Mh'
subhalos_smhm_ext['pq_func_var2[5]{2}'] = '1+z'

for i in range(0, 27):
    subhalos_smhm_ext['pq_func_par%i[5]{2}' % i] = setup['pq_func_par%i[0]{0}' % i]

subhalos_smhm_ext['pop_fstar{3}'] = 'link:fstar:2'

##
# Allows subhalos to have different SFR than centrals
subhalos_sfr_ext = {}
subhalos_sfr_ext['pop_sfr{2}'] = 'pq[6]'
subhalos_sfr_ext['pq_func[6]{2}'] = 'dplx_evolB13'
subhalos_sfr_ext['pq_func_var[6]{2}'] = 'Mh'
subhalos_sfr_ext['pq_func_var2[6]{2}'] = '1+z'


for i in range(0, 27):
    subhalos_sfr_ext['pq_func_par%i[6]{2}' % i] = setup['pq_func_par%i[1]{0}' % i]

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

# Scaling relationships for common strong lines
# Each pair is rest wavelength [Angstroms] and L_line [erg/s/(Msun/yr)]
lines = {}
lines['pop_lum_per_sfr_at_wave{0}'] = \
    [
     (1216., 1.21e42),            # Ly-a
     (6563, 1.27e41),             # H-alpha
     (5007, 1.32e41),             # [O III]
     (4861, 0.44e41),             # H-beta
     (4340, 0.468 * 0.44e41),     # H-gamma
     (4102, 0.259 * 0.44e41),     # H-delta
     (3970, 0.159 * 0.44e41),     # H-epsilon
     (3727, 0.71e41),             # [O II]
     (1.87e4, 1.27e41 * 0.123),   # [P-alpha]
     (3.3e4, lsun * 10**6.6)]     # 3.3 micron PAH (Lai+ 2020)
lines['pop_lum_per_sfr_at_wave{2}'] = lines['pop_lum_per_sfr_at_wave{0}']

no_lines = \
{
 'pop_lum_per_sfr_at_wave{0}': None,
 'pop_lum_per_sfr_at_wave{2}': None,
}

faster = \
{
 "halo_dlogM": 0.05,
 "halo_tmin": 100,
 "halo_tmax": 13.7e3,
 "halo_dt": 100,
}

fast = \
{
 "halo_dlogM": 0.025,
 "halo_tmin": 100,
 "halo_tmax": 13.7e3,
 "halo_dt": 100,
}

slow = \
{
 "halo_dlogM": 0.01,
 "halo_tmin": 30,
 "halo_dt": 10,
}

# Lowest dimensional model we've got?
# Need to be careful with this
_base = \
{
'pq_func_par0[0]{0}': 6.1764e-05,
'pq_func_par1[0]{0}': 9.1754e+11,
'pq_func_par2[0]{0}': 1.4473e+00,
'pq_func_par3[0]{0}': -5.5587e-01,
'pq_func_par5[0]{0}': -1.0034e+00,
'pq_func_par6[0]{0}': 5.8955e-01,
'pq_func_par7[0]{0}': -6.7433e-01,
'pq_func_par8[0]{0}': 1.6365e-01,
'pq_func_par0[2]{0}': 2.4506e-01,
'pq_func_par1[2]{0}': 8.2420e-01,
'pq_func_par2[2]{0}': 1.2364e+01,
'pq_func_par3[2]{0}': -2.0167e-01,
'pq_func_par4[2]{0}': -1.0386e-01,
'pq_func_par8[2]{0}': 3.6803e-01,
'pq_func_par5[2]{0}': -6.5620e-01,
'pq_func_par9[2]{0}': 6.3437e-01,
'pq_func_par6[2]{0}': -2.3490e+00,
'pq_func_par10[2]{0}': 1.0774e+00,
'pq_func_par7[2]{0}': 5.3345e-01,
'pq_func_par11[2]{0}': -2.8394e-01,
'pq_func_par0[1]{0}': 4.2053e-04,
'pq_func_par1[1]{0}': 2.7720e+11,
'pq_func_par2[1]{0}': 2.3336e+00,
'pq_func_par3[1]{0}': 4.9890e-01,
'pq_func_par5[1]{0}': -2.8294e+00,
'pq_func_par9[1]{0}': 1.9135e+00,
'pq_func_par6[1]{0}': 2.5113e+00,
'pq_func_par10[1]{0}': -1.0102e+00,
'pq_func_par7[1]{0}': -5.7024e-01,
'pq_func_par11[1]{0}': -8.2598e-02,
'pq_func_par8[1]{0}': -1.2234e+00,
'pq_func_par12[1]{0}': 6.7375e-01,
'pq_func_par0[4]{0}': 1.1055e+00,
'pq_func_par2[4]{0}': 8.7996e-03,
'pq_func_par5[4]{0}': -2.3831e-01,
'pq_func_par6[4]{0}': 8.9493e-02,
'pop_scatter_sfh{0}': 1.4846e-01,
'pop_sfr_below_ms{1}': 1.4333e+03,
'pop_sys_mstell_now{0}': -4.2003e-02,
'pop_sys_mstell_a{0}': 9.3474e-02,
'pop_sys_sfr_now{0}': 2.1676e-01,
'pop_sys_sfr_a{0}': 1.3600e-02,
}

#setup = base.copy()
#base.update(_base)

sed_modeling = \
{
 'pop_lum_tab{0}': f"{HOME}/.ares/ares_ebl_data/ares_base_seds_acen_beta_0.hdf5",
 'pop_lum_tab{1}': f"{HOME}/.ares/ares_ebl_data/ares_base_seds_qcen_beta_0.hdf5",
 'pop_lum_tab{2}': f"{HOME}/.ares/ares_ebl_data/ares_base_seds_acen_beta_0.hdf5",
 'pop_lum_tab{3}': f"{HOME}/.ares/ares_ebl_data/ares_base_seds_qcen_beta_0.hdf5",
}

sys_b13 = \
{
 'pop_sys_method{0}': "b13",
 'pop_sys_method{1}': "b13",
 'pop_sys_method{2}': "b13",
 'pop_sys_method{3}': "b13",
}
