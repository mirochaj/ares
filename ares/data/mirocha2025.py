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
 'tau_clumpy': 1,     # 1 = all < 912A photons gone, 2 = all < 1216A gone, 
                      # can also set to 'madau1995' for more detailed model.

 'cosmology_id': 'best',
 'cosmology_name': 'planck_TTTEEE_lowl_lowE',
 'cosmological_Mmin': None,

 'first_light_redshift': 15,
 'final_redshift': 6e-3,

 'tau_redshift_bins': 100,

 'halo_dlnk': 0.05,
 'halo_lnk_min': -9.,
 'halo_lnk_max': 11.,

 #'interpolate_cosmology_in_z': True,
}

basic_settings = setup.copy()

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
 'pop_age': (100., 4e3),
 'pop_Z': (0.02, 0.02), # placeholder, really
 'pop_binaries': False,

 'pop_Tmin': None,
 'pop_Mmin': 1e8,
 'pop_Mmax': None,

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
 'pq_func_par4[1]': 1e10,   # Mh anchor
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
 'pq_func_par21[1]': 0.0,   # Turn-over mass
 'pq_func_par22[1]': 0.0,   # upturn
 'pq_func_par23[1]': 0.0,   # upturn
 'pq_func_par24[1]': 0.0,   # evolution in turn-over mass
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
centrals_q['pop_age'] = 5e3
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
ihl_scaled['pop_focc_inv'] = False
ihl_scaled['pop_fstar'] = 'link:fstar:0' # Use SF centrals' SMHM...for now
ihl_scaled['pop_age'] = 5e3
ihl_scaled['pop_ihl'] = 'pq[50]'
ihl_scaled['pq_func[50]'] = 'pl_evolN'
ihl_scaled['pq_func_var[50]'] = 'Mh'
ihl_scaled['pq_func_var2[50]'] = '1+z'
ihl_scaled['pq_func_par0[50]'] = 0.01 # 1% of stellar mass -> IHL
ihl_scaled['pq_func_par1[50]'] = 1e12
ihl_scaled['pq_func_par2[50]'] = 1.  # Linear Mh dependence
ihl_scaled['pq_func_par3[50]'] = 1.  # Anchored to z=0
ihl_scaled['pq_func_par4[50]'] = 0   # No evolution by default [illustrative]
ihl_scaled['pq_val_ceil[50]'] = 0.7

# Deterministic luminosity 
ihl_scaled['pop_scatter_sfh{4}'] = 0


ihl_scaled['pop_include_1h'] = True
ihl_scaled['pop_include_2h'] = True
ihl_scaled['pop_include_shot'] = False
ihl_scaled['pop_Mmin'] = 1e10
#ihl_scaled['pop_Mmax'] = 1e15
ihl_scaled['pop_Tmin'] = None


# These numbers are Purcell-like
ihl_tanh = ihl_scaled.copy()
ihl_tanh['pq_func[50]'] = 'logtanh_abs'
ihl_tanh['pq_func_par0[50]'] = 0.7
ihl_tanh['pq_func_par1[50]'] = 0.0
ihl_tanh['pq_func_par2[50]'] = 13.6
ihl_tanh['pq_func_par3[50]'] = -1.
ihl_tanh['pq_val_ceil[50]'] = 0.7

ihl_tanh_zevol = ihl_tanh.copy()

#ihl_b19 = ihl_scaled.copy()
#ihl_b19['pq_func_par0[50]'] = 0.01
#ihl_b19['pq_func_par1[50]'] = 1e12
#ihl_b19['pq_func_par2[50]'] = 0.7
#ihl_b19['pq_val_ceil[50]'] = 0.99
#ihl_b19['pq_val_floor[50]{4}'] = 3e-3

ihl_p24 = ihl_scaled.copy()
ihl_p24['pq_func_par0[50]'] = 0.13
ihl_p24['pq_func_par1[50]'] = 1e12
ihl_p24['pq_func_par2[50]'] = 0.5
ihl_p24['pq_val_ceil[50]'] = 0.7

ihl_c24 = ihl_scaled.copy()
ihl_c24['pq_func_par0[50]'] = 0.11
ihl_c24['pq_func_par1[50]'] = 1e12
ihl_c24['pq_func_par2[50]'] = 0.25
ihl_c24['pq_val_ceil[50]'] = 0.7

ihl_p07 = ihl_tanh.copy()
ihl_p07['pq_func_par0[50]'] = 0.7
ihl_p07['pq_func_par1[50]'] = 0.0
ihl_p07['pq_func_par2[50]'] = 13.6
ihl_p07['pq_func_par3[50]'] = -1.
ihl_p07['pq_val_ceil[50]'] = 0.7  
ihl_p07['pq_val_floor[50]'] = 0.

ihl_b19 = ihl_tanh.copy()
ihl_b19['pq_func_par0[50]'] = 0.7
ihl_b19['pq_func_par1[50]'] = 3e-3
ihl_b19['pq_func_par2[50]'] = 14.1
ihl_b19['pq_func_par3[50]'] = -0.8
ihl_b19['pq_val_ceil[50]'] = 0.7

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
satellites_sf['pop_scatter_sfh'] = 'pop_scatter_sfh{0}'

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
satellites_q['pop_fstar'] = 'link:fstar:1'
satellites_q['pop_ssfr'] = None
#satellites_q['pop_scatter_sfh'] = 'pop_scatter_sfh{0}'
#satellites_q['pop_scatter_smhm'] = 'pop_scatter_smhm{1}'

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
dust_x['pq_func[40]{0}'] = 'pl_evolB13'
dust_x['pq_func_var[40]{0}'] = 'wave'
dust_x['pq_func_var2[40]{0}'] = '1+z'
dust_x['pq_func_par0[40]{0}'] = 1
dust_x['pq_func_par1[40]{0}'] = 5500
dust_x['pq_func_par2[40]{0}'] = 0.0
dust_x['pq_func_par3[40]{0}'] = 0     # norm
dust_x['pq_func_par4[40]{0}'] = 0     # slope
dust_x['pq_func_par5[40]{0}'] = 0     # norm
dust_x['pq_func_par6[40]{0}'] = 0     # slope
dust_x['pq_func_par7[40]{0}'] = 0     # norm
dust_x['pq_func_par8[40]{0}'] = 0     # slope
dust_x['pq_func_par9[40]{0}'] = 0     # slope
dust_x['pq_func_par10[40]{0}'] = 0     # slope

no_dust = {'pop_dust_template{0}': None, 'pop_Av{0}': 0}

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

dust_dplx = \
{
 'pq_func[4]{0}': 'dplx_evolB13',
 'pq_func_var[4]{0}': 'Mh',
 'pq_func_var2[4]{0}': '1+z',
 'pq_func_par0[4]{0}': 0.0,
 'pq_func_par1[4]{0}': 1e12,
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
  # Extension!
 'pq_func_par21[4]{0}': 5.0, # evolution done in log10(Mturn), hence default > 0
 'pq_func_par22[4]{0}': 0.0,
 'pq_func_par23[4]{0}': 0.0,
 'pq_func_par24[4]{0}': 0.0,
 'pq_func_par25[4]{0}': 0.0,
 'pq_func_par26[4]{0}': 0.0,
}

dust_linlog = \
{
 'pq_func[4]{0}': 'linlog_evolB13',
 'pq_func_var[4]{0}': 'Ms',
 'pq_func_var2[4]{0}': '1+z',
 'pq_func_par0[4]{0}': 0.5,
 'pq_func_par1[4]{0}': 10,             # log10(Mstell/Msun) we pin to
 'pq_func_par2[4]{0}': 0.1,            # slope
 # Start evol params
 'pq_func_par3[4]{0}': 0.,             # norm  (1 - a)
 'pq_func_par4[4]{0}': 0,              # slope (1 - a)
 'pq_func_par5[4]{0}': 0,              # norm  log(1+z)
 'pq_func_par6[4]{0}': 0,              # slope log(1+z)
 'pq_func_par7[4]{0}': 0,              # norm  z
 'pq_func_par8[4]{0}': 0,              # slope z
 'pq_func_par9[4]{0}': 0.0,            # norm  a
 'pq_func_par10[4]{0}': 0.0,           # slope a
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
ihl['pop_sed{4}'] = 'bc03_2013'
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

setup_centrals = setup.copy()
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
     (3.28e4, lsun * 10**6.6)]     # 3.3 micron PAH (Lai+ 2020)
lines['pop_lum_per_sfr_at_wave{2}'] = lines['pop_lum_per_sfr_at_wave{0}']

lines_wprof = {}
lines_wprof['pop_lum_per_sfr_at_wave{0}'] = \
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
     (3.28e4, 0.505 * lsun * 10**6.6, 0.0301e4),     # 3.3 micron PAH (Lai+ 2020)
     (3.28e4, 0.495 * lsun * 10**6.6, 0.1028e4),
     (3.40e4, 0.08592 * lsun * 10**6.6, 0.0301e4),
     (3.48e4, 0.17205 * lsun * 10**6.6, 0.0555e4)]

lines_wprof['pop_lum_per_sfr_at_wave{2}'] = lines_wprof['pop_lum_per_sfr_at_wave{0}']

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

very_slow = \
{
 "halo_dlogM": 0.01,
 "halo_tmin": 30,
 "halo_dt": 1,
}

# Lowest dimensional model we've got?
# Need to be careful with this
_base = \
{
'pq_func_par0[0]{0}': 1.3244e-04, 
'pq_func_par1[0]{0}': 1.5496e+12, 
'pq_func_par2[0]{0}': 1.2667e+00, 
'pq_func_par3[0]{0}': -8.1479e-01, 
'pq_func_par0[10]{1}': 1.0786e-05, 
'pq_func_par1[10]{1}': 9.1823e+11, 
'pq_func_par2[10]{1}': 1.6750e+00, 
'pq_func_par3[10]{1}': -3.4657e-01, 
'pq_func_par5[0]{0}': 6.6381e-01, 
'pq_func_par9[0]{0}': -9.0677e-01, 
'pq_func_par6[0]{0}': 4.7558e-01, 
'pq_func_par10[0]{0}': 5.5495e-01, 
'pq_func_par7[0]{0}': -1.1704e+00, 
'pq_func_par11[0]{0}': 1.1723e-01, 
'pq_func_par8[0]{0}': 1.1042e+00, 
'pq_func_par12[0]{0}': -4.5759e-01, 
'pq_func_par5[10]{1}': 1.4652e+00, 
'pq_func_par9[10]{1}': -1.6893e+00, 
'pq_func_par6[10]{1}': 2.3200e-02, 
'pq_func_par10[10]{1}': 3.2256e-01, 
'pq_func_par7[10]{1}': -4.8524e-01, 
'pq_func_par11[10]{1}': 4.3121e+00, 
'pq_func_par8[10]{1}': -4.4594e-01, 
'pq_func_par12[10]{1}': 9.3981e-01, 
'pq_func_par0[2]{0}': 4.3086e-01, 
'pq_func_par1[2]{0}': 9.7883e-01, 
'pq_func_par2[2]{0}': 1.2274e+01, 
'pq_func_par3[2]{0}': -2.8718e-01, 
'pq_func_par4[2]{0}': -2.3810e-01, 
'pq_func_par8[2]{0}': -1.7166e+00, 
'pq_func_par5[2]{0}': -2.6325e+00, 
'pq_func_par9[2]{0}': 9.2646e-01, 
'pq_func_par6[2]{0}': -3.2199e-01, 
'pq_func_par10[2]{0}': 2.4950e-01, 
'pq_func_par7[2]{0}': 1.3614e+00, 
'pq_func_par11[2]{0}': -6.3212e-01, 
'pq_func_par0[1]{0}': 7.3595e-04, 
'pq_func_par1[1]{0}': 5.9922e+11, 
'pq_func_par2[1]{0}': 1.9341e+00, 
'pq_func_par3[1]{0}': -2.2378e-02, 
'pq_func_par5[1]{0}': 1.9446e+00, 
'pq_func_par9[1]{0}': 5.0245e-01, 
'pq_func_par6[1]{0}': 3.6384e-01, 
'pq_func_par10[1]{0}': -4.5445e-01, 
'pq_func_par7[1]{0}': 4.6274e-01, 
'pq_func_par11[1]{0}': -2.6692e-01, 
'pq_func_par8[1]{0}': 6.6447e-01, 
'pq_func_par12[1]{0}': 1.3774e-01, 
'pq_func_par0[4]{0}': 9.1181e-01, 
'pq_func_par1[4]{0}': 1.6436e+12, 
'pq_func_par2[4]{0}': 1.3054e-02, 
'pq_func_par3[4]{0}': -9.1258e-01, 
'pq_func_par5[4]{0}': 7.6745e-01, 
'pq_func_par9[4]{0}': -1.9899e-01, 
'pq_func_par6[4]{0}': -7.5186e-01, 
'pq_func_par10[4]{0}': 1.4085e-01, 
'pq_func_par7[4]{0}': 2.1244e+00, 
'pq_func_par11[4]{0}': -7.6331e-01, 
'pq_func_par8[4]{0}': 7.7998e-01, 
'pq_func_par12[4]{0}': 1.3016e-01, 
'pop_scatter_sfh{0}': 3.5867e-01, 
'pop_sfr_below_ms{1}': 8.4874e+01, 
'pop_sys_mstell_now{0}': -2.0935e-01, 
'pop_sys_mstell_a{0}': 2.0424e-01, 
'pop_sys_sfr_now{0}': 9.4813e-03, 
'pop_sys_sfr_a{0}': 1.7007e-02, 
}

_base_smhm_univ = \
{
'pq_func_par0[0]{0}': 0.00036576972411987284,
 'pq_func_par1[0]{0}': 3586042344255.609,
 'pq_func_par2[0]{0}': 1.0690516910828674,
 'pq_func_par3[0]{0}': -0.8683520472189735,
 'pq_func_par0[10]{1}': 2.4537458934777302e-06,
 'pq_func_par1[10]{1}': 3423628037955.847,
 'pq_func_par2[10]{1}': 1.6434698264070247,
 'pq_func_par3[10]{1}': -0.4733780759504128,
 'pq_func_par0[2]{0}': 0.19912659563862484,
 'pq_func_par1[2]{0}': 0.9079166445906448,
 'pq_func_par2[2]{0}': 12.295770966429185,
 'pq_func_par3[2]{0}': -0.1327008519287325,
 'pq_func_par4[2]{0}': -4.639056446096568,
 'pq_func_par8[2]{0}': -0.1410681180954496,
 'pq_func_par5[2]{0}': -2.0089755644145693,
 'pq_func_par9[2]{0}': 0.9828167246323941,
 'pq_func_par6[2]{0}': -0.5344995205096565,
 'pq_func_par10[2]{0}': 0.39167747042370904,
 'pq_func_par7[2]{0}': 0.5026449692594634,
 'pq_func_par11[2]{0}': -0.26101109136692957,
 'pq_func_par0[1]{0}': 0.0008233525187367075,
 'pq_func_par1[1]{0}': 146999154470.56808,
 'pq_func_par2[1]{0}': 2.54526789136891,
 'pq_func_par3[1]{0}': 0.7737154573134529,
 'pq_func_par5[1]{0}': -2.653825334571936,
 'pq_func_par9[1]{0}': 1.523091483007643,
 'pq_func_par6[1]{0}': 2.5077317578238354,
 'pq_func_par10[1]{0}': -0.7093149884797,
 'pq_func_par7[1]{0}': -0.35668875527315697,
 'pq_func_par11[1]{0}': -0.2986135081605227,
 'pq_func_par8[1]{0}': -4.320736441708885,
 'pq_func_par12[1]{0}': 2.0618527004109737,
 'pq_func_par0[4]{0}': 1.2817054222133069,
 'pq_func_par1[4]{0}': 993540967796.7826,
 'pq_func_par2[4]{0}': 0.03298877241207199,
 'pq_func_par3[4]{0}': -0.6800918415574786,
 'pq_func_par5[4]{0}': 0.4561503842789619,
 'pq_func_par9[4]{0}': -0.10749626733006643,
 'pq_func_par6[4]{0}': -1.3814962026103803,
 'pq_func_par10[4]{0}': 1.013289813196224,
 'pq_func_par7[4]{0}': 2.544550329691066,
 'pq_func_par11[4]{0}': -0.9819717814214239,
 'pq_func_par8[4]{0}': -2.8234501553108116,
 'pq_func_par12[4]{0}': 1.643708817087465,
 'pop_scatter_sfh{0}': 0.44007987837757934,
 'pop_sfr_below_ms{1}': 182.33182699547618,
 'pop_sys_mstell_now{0}': -0.28159237932347375,
 'pop_sys_mstell_a{0}': 0.09152500543348921,
 'pop_sys_sfr_now{0}': 0.0013009416539419326,
 'pop_sys_sfr_a{0}': 0.008643024161763914
}

_base_smhm_evol = \
{'pq_func_par0[0]{0}': 0.00024340818505629087,
 'pq_func_par1[0]{0}': 4432643085721.096,
 'pq_func_par2[0]{0}': 1.2711114162862631,
 'pq_func_par3[0]{0}': -0.3051492520095239,
 'pq_func_par0[10]{1}': 4.843669332479228e-06,
 'pq_func_par1[10]{1}': 2014695744144.7297,
 'pq_func_par2[10]{1}': 1.6439795830453647,
 'pq_func_par3[10]{1}': -0.35002344629511645,
 'pq_func_par5[0]{0}': -0.983444639361645,
 'pq_func_par9[0]{0}': -1.1181339245014636,
 'pq_func_par6[0]{0}': -4.1951081806445645,
 'pq_func_par10[0]{0}': 4.37791004560655,
 'pq_func_par7[0]{0}': -0.7905597384245008,
 'pq_func_par11[0]{0}': 0.08295044078231055,
 'pq_func_par8[0]{0}': -0.922018537767296,
 'pq_func_par12[0]{0}': -0.4427307454247793,
 'pq_func_par5[10]{1}': 2.470369901568695,
 'pq_func_par9[10]{1}': -1.4103107097822063,
 'pq_func_par6[10]{1}': -1.049867095195406,
 'pq_func_par10[10]{1}': 0.6759217791847014,
 'pq_func_par7[10]{1}': -4.364305220725775,
 'pq_func_par11[10]{1}': 1.689412895264865,
 'pq_func_par8[10]{1}': -1.6527422568397754,
 'pq_func_par12[10]{1}': 0.9552332857980068,
 'pq_func_par0[2]{0}': 0.07462240913649396,
 'pq_func_par1[2]{0}': 0.6452348627356916,
 'pq_func_par2[2]{0}': 12.143665331444609,
 'pq_func_par3[2]{0}': -0.1483955059139539,
 'pq_func_par4[2]{0}': -0.21853035459435977,
 'pq_func_par8[2]{0}': -1.5867957426130983,
 'pq_func_par5[2]{0}': 1.0717050592683606,
 'pq_func_par9[2]{0}': -0.30126731858101663,
 'pq_func_par6[2]{0}': 0.18793307768908268,
 'pq_func_par10[2]{0}': 0.1801743223797549,
 'pq_func_par7[2]{0}': -0.3008877981810485,
 'pq_func_par11[2]{0}': -0.0451121696867538,
 'pq_func_par0[1]{0}': 0.0010960294500633263,
 'pq_func_par1[1]{0}': 305470827115.1007,
 'pq_func_par2[1]{0}': 2.2557455352497713,
 'pq_func_par3[1]{0}': 1.0213159635888145,
 'pq_func_par5[1]{0}': -3.938487654901255,
 'pq_func_par9[1]{0}': 2.4185488519201197,
 'pq_func_par6[1]{0}': 2.633566748148678,
 'pq_func_par10[1]{0}': -0.992813961284668,
 'pq_func_par7[1]{0}': -0.5750272876212722,
 'pq_func_par11[1]{0}': 0.0050600294649953415,
 'pq_func_par8[1]{0}': -1.8978075443855054,
 'pq_func_par12[1]{0}': 0.6822435303576575,
 'pq_func_par0[4]{0}': 0.4386717065809691,
 'pq_func_par1[4]{0}': 3014273283275.125,
 'pq_func_par2[4]{0}': 0.22772419699833676,
 'pq_func_par3[4]{0}': -0.7773397105141043,
 'pq_func_par5[4]{0}': 0.2748510912874824,
 'pq_func_par9[4]{0}': -0.13699053531988065,
 'pq_func_par6[4]{0}': 2.9907271775153705,
 'pq_func_par10[4]{0}': -1.2065527459224699,
 'pq_func_par7[4]{0}': 2.7374686670988417,
 'pq_func_par11[4]{0}': -2.2971330480428045,
 'pq_func_par8[4]{0}': 2.7291653747149787,
 'pq_func_par12[4]{0}': -0.645307009351606,
 'pop_scatter_sfh{0}': 0.41137693427830385,
 'pop_sfr_below_ms{1}': 64.61353557557199,
 'pop_sys_mstell_now{0}': -0.24066444958980104,
 'pop_sys_mstell_a{0}': 0.14437770641779976,
 'pop_sys_sfr_now{0}': 0.0014440131161852915,
 'pop_sys_sfr_a{0}': 0.01464925069842566,
}

sed_modeling_univ = \
{
 'pop_lum_tab{0}': f"{HOME}/.ares/sedtabs/sedtab_pop_0_smhm_univ_best.hdf5",
 'pop_lum_tab{1}': f"{HOME}/.ares/sedtabs/sedtab_pop_1_smhm_univ_best.hdf5",
 'pop_lum_tab{2}': f"{HOME}/.ares/sedtabs/sedtab_pop_0_smhm_univ_best.hdf5",
 'pop_lum_tab{3}': f"{HOME}/.ares/sedtabs/sedtab_pop_1_smhm_univ_best.hdf5",
}

sed_modeling_evol = \
{
 'pop_lum_tab{0}': f"{HOME}/.ares/sedtabs/sedtab_pop_0_smhm_evol_best.hdf5",
 'pop_lum_tab{1}': f"{HOME}/.ares/sedtabs/sedtab_pop_1_smhm_evol_best.hdf5",
 'pop_lum_tab{2}': f"{HOME}/.ares/sedtabs/sedtab_pop_0_smhm_evol_best.hdf5",
 'pop_lum_tab{3}': f"{HOME}/.ares/sedtabs/sedtab_pop_1_smhm_evol_best.hdf5",
}


no_sed_modeling = \
{
 'pop_lum_tab{0}': None,
 'pop_lum_tab{1}': None,
 'pop_lum_tab{2}': None,
 'pop_lum_tab{3}': None,
}

sys_b13 = \
{
 'pop_sys_method{0}': "b13",
 'pop_sys_method{1}': "b13",
 'pop_sys_method{2}': "b13",
 'pop_sys_method{3}': "b13",
}

scatter_flex = \
{
 'pop_scatter_sfh{0}': 0,
 'pop_scatter_sfh{1}': 0,
 'pop_scatter_sfh{2}': 0,
 'pop_scatter_sfh{3}': 0,
 'pop_scatter_sfr{0}': 0.,
 'pop_scatter_smhm{0}': 0.,
 'pop_scatter_smhm{1}': 0.,
}

# 'base' model has:
# (i) different SMHM for star-forming and quiescent sources
# (ii) DPL SFR-Mh relation
# (iii) DPL Dust-Mh relation
# (iv) systematics not identical to B13
# (v) satellites == centrals at given (sub)halo mass

#base = setup.copy()
#base.update(smhm_Q)
#base.update(dust_dplx)
#base.update(_base)
#base.update(sed_modeling)
#base.update(lines_wprof)

smhm_univ = setup.copy()
smhm_univ.update(smhm_Q)
smhm_univ.update(dust_dplx)
smhm_univ.update(_base_smhm_univ)
smhm_univ.update(sed_modeling_univ)
smhm_univ.update(lines_wprof)

smhm_evol = setup.copy()
smhm_evol.update(smhm_Q)
smhm_evol.update(dust_dplx)
smhm_evol.update(_base_smhm_evol)
smhm_evol.update(sed_modeling_evol)
smhm_evol.update(lines_wprof)