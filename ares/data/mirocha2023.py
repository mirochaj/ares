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
 'pop_binaries': False,
 'pop_rad_yield': 'from_sed',

 'pop_fesc': 0.2,
 'pop_sed_degrade': 10,

 'pop_nebular': 0,

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
 'pq_func[2]': 'logtanh_abs_evolMFC', # Evolving midpoint, floor, ceiling
 'pq_func_var[2]': 'Mh',
 'pq_func_var2[2]': '1+z',
 'pq_val_ceil[2]': 1,
 'pq_func_par0[2]': 1,
 'pq_func_par1[2]': 0.0,
 'pq_func_par2[2]': 11.8,
 'pq_func_par3[2]': 0.6,
 'pq_func_par4[2]': 0.1,
 'pq_func_par5[2]': 1,
 'pq_func_par6[2]': 0,
 'pq_func_par7[2]': 0,

}

#centrals_sf.update(_base)

centrals_q = centrals_sf.copy()
centrals_q['pop_ssfr'] = None
centrals_q['pop_ssp'] = True
centrals_q['pop_age'] = 3e3
centrals_q['pop_Z'] = 0.02
centrals_q['pop_fstar'] = 'link:fstar:0'
centrals_q['pop_focc'] = 'link:focc:0'
centrals_q['pop_nebular'] = 0

centrals_sf_old = centrals_q.copy()

# Add this later so old component of SFGs keeps original focc
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
satellites_sf['pop_centrals'] = 0
satellites_sf['pop_centrals_id'] = 0
satellites_sf['pop_prof_1h'] = 'nfw'
satellites_sf['pop_include_1h'] = True
satellites_sf['pop_include_2h'] = True
satellites_sf['pop_include_shot'] = False
satellites_sf['pop_fstar'] = 'link:fstar:0'
satellites_sf['pop_ssfr'] = 'link:ssfr:0'

satellites_old = centrals_sf_old.copy()
satellites_old['pop_centrals'] = 0
satellites_old['pop_centrals_id'] = 0
satellites_old['pop_prof_1h'] = 'nfw'
satellites_old['pop_include_1h'] = True
satellites_old['pop_include_2h'] = True
satellites_old['pop_include_shot'] = False
satellites_old['pop_fstar'] = 'link:fstar:0'
satellites_old['pop_ssfr'] = None
satellites_old['pop_focc'] = 1

satellites_all = satellites_sf.copy()
satellites_all['pop_focc'] = 1

#
ihl_from_sat = centrals_sf_old.copy()
ihl_from_sat['pop_focc'] = 1
ihl_from_sat['pop_centrals'] = 0
ihl_from_sat['pop_centrals_id'] = 0
ihl_from_sat['pop_prof_1h'] = 'nfw'
ihl_from_sat['pop_fsurv'] = 'link:fsurv:3'
ihl_from_sat['pop_surv_inv'] = True
ihl_from_sat['pop_include_1h'] = True
ihl_from_sat['pop_include_2h'] = True
ihl_from_sat['pop_include_shot'] = False

base = _base.copy()
_pop0 = centrals_sf.copy()
_pop1 = centrals_sf_old.copy()
_pop2 = centrals_q.copy()
_pop3 = satellites_all.copy()
_pop4 = satellites_old.copy()
_pop5 = ihl_from_sat.copy()

for i, _pop in enumerate([_pop0, _pop1, _pop2]):
    pf = {}
    for par in _pop.keys():
        pf[par + '{%i}' % i] = _pop[par]

    base.update(pf)

subhalos = {}
for i, _pop in enumerate([_pop3, _pop4, _pop5]):
    pf = {}
    for par in _pop.keys():
        pf[par + '{%i}' % i] = _pop[par]

    subhalos.update(pf)

subhalos['pop_fsurv{3}'] = 'pq[3]'
subhalos['pq_func[3]{3}'] = 'logtanh_abs_evolM'
subhalos['pq_func_var[3]{3}'] = 'Mh'
subhalos['pq_func_var2[3]{3}'] = '1+z'
subhalos['pq_val_ceil[3]{3}'] = 1
subhalos['pq_func_par0[3]{3}'] = 0.0  # step = par0-par1
subhalos['pq_func_par1[3]{3}'] = 0.95 # fsurv = par1 + step * tanh(stuff)
subhalos['pq_func_par2[3]{3}'] = 11
subhalos['pq_func_par3[3]{3}'] = 0.7 # dlogM
subhalos['pq_func_par4[3]{3}'] = 0.  # Evolution in midpoint
subhalos['pq_func_par5[3]{3}'] = 1   # Pin to z=0

subhalos['pop_fsurv{4}'] = 'link:fsurv:3'

subhalos['pop_Mmin{3}'] = 1e10
subhalos['pop_Mmin{4}'] = 1e10
subhalos['pop_Mmin{5}'] = 1e10

subhalos['pop_fsurv{5}'] = 'link:fsurv:3'
subhalos['pop_fsurv_inv{5}'] = True

dust_template = {}
dust_template['pop_dustext_template'] = 'milkyway_rv4'
dust_template['pop_Av'] = 'pq[4]'
dust_template['pq_func[4]'] = 'pl_evolN'
dust_template['pq_func_var[4]'] = 'Ms'
dust_template['pq_func_var2[4]'] = '1+z'
dust_template['pq_func_par0[4]'] = 1
dust_template['pq_func_par1[4]'] = 1e10
dust_template['pq_func_par2[4]'] = 0.1
dust_template['pq_func_par3[4]'] = 1.  # Anchored to z=0
dust_template['pq_func_par4[4]'] = 0.  # no evolution yet.

dust = \
{
 'pop_dust_yield': 0.4,

 # Dust opacity vs. wavelength
 "pop_dust_absorption_coeff": 'pq[20]',   # opacity in [cm^2 / g]
 "pq_func[20]": 'pl',
 'pq_func_var[20]': 'wave',
 'pq_func_var_lim[20]': (0., np.inf),
 'pq_func_var_fill[20]': 0.0,
 'pq_func_par0[20]': 1e5,      # opacity at wavelength below
 'pq_func_par1[20]': 1e3,
 'pq_func_par2[20]': -1.,

 # Screen parameters
 'pop_dust_fcov': 1,
 "pop_dust_scale": 'pq[22]',       # Scale radius [in kpc]
 #"pq_func[22]": 'pl_evolN',
 #'pq_func_var[22]': 'Mh',
 #'pq_func_var2[22]': '1+z',
 #'pq_func_par0[22]': 1.6,     # Note that Rhalo ~ Mh^1/3 / (1+z)
 #'pq_func_par1[22]': 1e10,
 #'pq_func_par2[22]': 0.45,
 #'pq_func_par3[22]': 5.,
 #'pq_func_par4[22]': 0.,

 "pq_func[22]": 'dpl_evolN',
 'pq_func_var[22]': 'Mh',
 'pq_func_var2[22]': '1+z',
 'pq_func_par0[22]': 1.6,     # Normalization of length scale
 'pq_func_par1[22]': 3e11,    # normalize at Mh=1e10
 'pq_func_par2[22]': 0.45,    # low-mass sope
 'pq_func_par3[22]': 0.45,    # high-mass slope
 'pq_func_par4[22]': 1e10,    # peak mass
 'pq_func_par5[22]': 1.,      # pin to z=1
 'pq_func_par6[22]': 0.0      # no z evolution by default

}

metals = \
{
 'pop_metal_yield': 0.1,

 # Fraction of metals lost
 "pop_metal_loss": 'pq[30]',
 "pq_func[30]": 'pl_evolN',
 'pq_func_var[30]': 'Mh',
 'pq_func_var2[30]': '1+z',
 'pq_func_par0[30]': 0.,
 'pq_func_par1[30]': 1e12,
 'pq_func_par2[30]': 0.5,
 'pq_func_par3[30]': 1.,   # pin to z=0
 'pq_func_par4[30]': 0.,   # no evolution
 'pq_val_ceil[30]': 1,
 'pq_val_floor[30]': 0,

 "pop_mass_loss": 'pq[31]',
 "pq_func[31]": 'pl_evolN',
 'pq_func_var[31]': 'Mh',
 'pq_func_var2[31]': '1+z',
 'pq_func_par0[31]': 0.,
 'pq_func_par1[31]': 1e12,
 'pq_func_par2[31]': 0.5,
 'pq_func_par3[31]': 1.,   # pin to z=0
 'pq_func_par4[31]': 0.,   # no evolution
 'pq_val_ceil[31]': 1,
 'pq_val_floor[31]': 0,
}

mzr = \
{
 'pop_mzr': 'pq[30]',
 'pop_fox': 0.03,
 "pq_func[30]": 'linear_evolN',
 'pq_func_var[30]': 'Ms',
 'pq_func_var2[30]': '1+z',
 'pq_func_par0[30]': 8.65,
 'pq_func_par1[30]': 10,
 'pq_func_par2[30]': 0.25,
 'pq_func_par3[30]': 1.,   # pin to z=0
 'pq_func_par4[30]': -0.08,   # no evolution
 'pq_val_ceil[30]': 9,
 'pq_val_floor[30]': 6,
}

gas = \
{
 "pop_gas_fraction": 'pq[31]',
 "pq_func[31]": 'pl_evolN',
 'pq_func_var[31]': 'Mh',
 'pq_func_var2[31]': '1+z',
 'pq_func_par0[31]': 0.7,
 'pq_func_par1[31]': 1e12,
 'pq_func_par2[31]': -0.25,
 'pq_func_par3[31]': 1.,   # pin to z=0
 'pq_func_par4[31]': 0.,   # no evolution
 'pq_val_ceil[31]': 1,
 'pq_val_floor[31]': 0,
}

Av = \
{
 "pop_Av": 'pq[31]',
 "pq_func[31]": 'pl_evolN',
 'pq_func_var[31]': 'Ms',
 'pq_func_var2[31]': '1+z',
 'pq_func_par0[31]': 0.5,
 'pq_func_par1[31]': 1e10,
 'pq_func_par2[31]': 0.25,
 'pq_func_par3[31]': 1.,   # pin to z=0
 'pq_func_par4[31]': 0.,   # no evolution
 'pq_val_floor[31]': 0,

 # Dust opacity vs. wavelength
 "pop_dust_absorption_coeff": 'pq[20]',   # opacity in [cm^2 / g]
 "pq_func[20]": 'pl',
 'pq_func_var[20]': 'wave',
 'pq_func_var_lim[20]': (0., np.inf),
 'pq_func_var_fill[20]': 0.0,
 'pq_func_par0[20]': 1e5,      # opacity at wavelength below
 'pq_func_par1[20]': 1e3,
 'pq_func_par2[20]': -1.,
}
