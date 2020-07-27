import numpy as np
from ares.physics.Constants import E_LyA, E_LL

_base = \
{
 # SFE
 'pop_fstar': 'pq[0]',
 'pq_func[0]': 'dpl_evolNP',
 'pq_func_var[0]': 'Mh',
 'pq_func_var2[0]': '1+z',

 # NIRB
 'tau_approx': 'neutral',
 'tau_clumpy': 'madau1995',

 # DPL in Mh: same base parameters as M17
 'pq_func_par0[0]': 0.03, # adjust peak-norm
 'pq_func_par1[0]': 2.8e11,
 'pq_func_par2[0]': 0.49,
 'pq_func_par3[0]': -0.61,
 'pq_func_par4[0]': 1e10,
 'pq_func_par5[0]': 5.,    # 1+z pivot
 'pq_func_par6[0]': 0.0,   # norm
 'pq_func_par7[0]': 0.0,   # Mp
 'pq_func_par8[0]': 0.0,   # Only use if slopes evolve, e.g., in dplp_evolNPS
 'pq_func_par9[0]': 0.0,   # Only use if slopes evolve, e.g., in dplp_evolNPS
 'pq_val_ceil[0]': 1.,     # Cap SFE at <= 1
}

_sed_updates = \
{
 'pop_sfr_model': 'ensemble',

 # Spectral synthesis
 'pop_sed': 'eldridge2009',
 'pop_binaries': False,
 'pop_rad_yield': 'from_sed',
 'pop_Emin': E_LyA,
 'pop_Emax': 24.6,
 'pop_fesc': 0.2,

 'pop_sed_degrade': 10,
 'pop_thin_hist': 10,
 'pop_aging': True,
 'pop_ssp': True,
 'pop_calib_lum': None,
 'pop_Z': 0.004,
 'pop_zdead': 3.5,

 # Synthesis control
 'pop_mag_bin': 1.0,            # Will bin to this mag-resolution for LF
 'pop_synth_cache_level': 0,    # 1 = more careful = slower
 'pop_synth_minimal': True,
 'pop_synth_zmax': 20.,
 'pop_Tmin': 1e4,               # Just for sake of SFRD

 # Metallicity evolution!?
 'pop_enrichment': False,
 'pop_metal_yield': 0.1,
 'pop_mass_yield': 0.15,
 'pop_fpoll': 1,

 # For reproducibility.
 'pop_dust_scatter_seed': 87112948,
 'pop_fduty_seed': 982323505,
}

_halo_updates = \
{
 # Use constant timestep
 'hmf_dt': 1.,
 'hmf_tmax': 2e3,
 'hmf_model': 'Tinker10',

 # Need to build enough halo histories at early times to get massive
 # halos at late times.
 'hgh_dlogM': 0.1,
 'hgh_Mmax': 10,

 # Add scatter to SFRs
 'pop_scatter_mar': 0.3,

 # For reproducibility.
 'pop_scatter_mar_seed': 10620202,

 # Use cosmology consistent with Paul's simulations.
 'cosmology_name': 'user',
 'cosmology_id': 'paul',
 "sigma_8": 0.8159,
 'primordial_index': 0.9652,
 'omega_m_0': 0.315579,
 'omega_b_0': 0.0491,
 'hubble_0': 0.6726,
 'omega_l_0': 1. - 0.315579,
}

_base.update(_sed_updates)
_base.update(_halo_updates)

_legacy = _base.copy()
_legacy['pop_sfr_model'] = 'sfe-func'
_legacy['pop_dust_yield'] = None
_legacy['pop_scatter_mar'] = 0
_legacy['pop_sed_degrade'] = None
_legacy['pop_thin_hist'] = 1
_legacy['pop_aging'] = False
_legacy['pop_ssp'] = False

# Dust-free, zcal=6
_legacy_best = \
{
 "pq_func_par0[0]": 0.020987095422405785,
 "pq_func_par1[0]": 154539759172.42554,
 "pq_func_par2[0]": 0.6320321585761033,
 "pq_func_par3[0]": -0.5073538305483938,
}

legacy = _legacy.copy()
legacy.update(_legacy_best)

legacy_irxb = legacy.copy()
legacy_irxb['dustcorr_method'] = 'meurer1999'
legacy_irxb['dustcorr_beta'] = 'bouwens2014'

# zcal=4
_legacy_irxb_best = \
{
 "pq_func_par0[0]": 0.019615994530922623,
 "pq_func_par1[0]": 168891637904.2078,
 "pq_func_par2[0]": 1.0521760410145145,
 "pq_func_par3[0]": -0.39520889465712994,
}

legacy_irxb.update(_legacy_irxb_best)

_screen = \
{
 'pop_dust_yield': 0.4,

 # Dust opacity vs. wavelength
 "pop_dust_kappa": 'pq[20]',   # opacity in [cm^2 / g]
 "pq_func[20]": 'pl',
 'pq_func_var[20]': 'wave',
 'pq_func_var_lim[20]': (912., np.inf),
 'pq_func_var_fill[20]': 0.0,
 'pq_func_par0[20]': 1e5,      # opacity at wavelength below
 'pq_func_par1[20]': 1e3,
 'pq_func_par2[20]': -1.,

 # Screen parameters
 'pop_dust_fcov': 1,
 "pop_dust_scale": 'pq[22]',       # Scale radius [in kpc]
 "pq_func[22]": 'pl_evolN',
 'pq_func_var[22]': 'Mh',
 'pq_func_var2[22]': '1+z',
 'pq_func_par0[22]': 1.6,     # Note that Rhalo ~ Mh^1/3 / (1+z)
 'pq_func_par1[22]': 1e10,
 'pq_func_par2[22]': 0.45,
 'pq_func_par3[22]': 5.,
 'pq_func_par4[22]': 0.,

 # Scatter in dust column density
 "pop_dust_scatter": 'pq[33]',
 'pq_func[33]': 'pl_evolN',
 'pq_func_var[33]': 'Mh',
 'pq_func_var2[33]': '1+z',
 'pq_func_par0[33]': 0.,    # No scatter by default
 'pq_func_par1[33]': 1e10,
 'pq_func_par2[33]': 0.,
 'pq_func_par3[33]': 5.,
 'pq_func_par4[33]': 0.,
}

_screen_dpl = \
{
 "pq_func[22]": 'dpl_evolN',
 'pq_func_par0[22]': 1.6,     # Normalization of length scale
 'pq_func_par1[22]': 3e11,    # normalize at Mh=1e10
 'pq_func_par2[22]': 0.45,    # low-mass sope
 'pq_func_par3[22]': 0.45,    # high-mass slope
 'pq_func_par4[22]': 1e10,    # peak mass
 'pq_func_par5[22]': 5.,      # pin to z=4
 'pq_func_par6[22]': 0.0      # no z evolution by default
}

plrd = _base.copy()
plrd.update(_screen)

# Just energy-regulated model for now with dust scale length going
# like the virial radius.
_evol_ereg_plrd = \
{
'pq_func_par2[0]': 0.66666,
'pq_func_par6[0]': 1.0,
'pq_func_par2[22]': 0.33333,
'pq_func_par4[22]': -1.,
}

_evol_ereg_dplrd = \
{
'pq_func_par2[0]': 0.66666,
'pq_func_par6[0]': 1.0,
'pq_func_par2[22]': 0.33333,
'pq_func_par3[22]': 0.33333,
'pq_func_par6[22]': -1.,
}

_evol_mreg_plrd = \
{
'pq_func_par2[0]': 0.33333,
'pq_func_par6[0]': 0.5,
'pq_func_par2[22]': 0.33333,
'pq_func_par4[22]': -1.,
}

_evol_mreg_dplrd = \
{
'pq_func_par2[0]': 0.33333,
'pq_func_par6[0]': 0.5,
'pq_func_par2[22]': 0.33333,
'pq_func_par3[22]': 0.33333,
'pq_func_par6[22]': -1.,
}

# Add models for no dust and IRX-beta approaches
univ = plrd.copy()
univ.update(_screen_dpl)

_univ_best = \
{
 "pq_func_par0[0]": 0.05499018374423549,
 "pq_func_par1[0]": 143774319309.14636,
 "pq_func_par2[0]": 0.804721973406274,
 "pq_func_par3[0]": -0.5305963428767742,
 "pq_func_par0[22]": 1.1170032134924464,
 "pq_func_par2[22]": 0.09030113968480673,
 "pq_func_par3[22]": 0.6949382430744515,
 "pq_func_par1[22]": 1033034647905.178,
 "pq_func_par0[33]": 0.003255298852620223,
}

univ.update(_univ_best)

# Not yet implemented
_univ_nodust_best = {}

univ_nodust = univ.copy()
univ_nodust['pop_dust_yield'] = None
univ_nodust.update(_univ_nodust_best)

_peak_best = \
{
 "pq_func_par0[0]": 0.0248695915993,
 "pq_func_par1[0]": 4.90587733667e+11,
 "pq_func_par7[0]": -0.0569331018797,
 "pq_func_par3[0]": -1.12413838053,
 "pq_func_par9[0]": 0.180419416177,
 "pq_func_par0[22]": 1.01045060124,
 "pq_func_par2[22]": 0.68365346273,
 "pq_func_par3[22]": 0.300420364685,
 "pq_func_par1[22]": 6.2296335743e+11,
 "pq_func_par0[33]": 0.110077637149,
}

ereg_epeak = univ.copy()
ereg_epeak['pq_func[0]'] = 'dpl_evolNPS'
ereg_epeak.update(_peak_best)
ereg_epeak.update(_evol_ereg_dplrd)

mreg_epeak = ereg_epeak.copy()
mreg_epeak.update(_evol_mreg_dplrd)

# Only difference between `univ` and `evol` models is through
# changes to evolution parameters.

_fduty = \
{
 'pop_fduty': 'pq[40]',
 "pq_func[40]": 'pl_evolN',
 'pq_func_var[40]': 'Mh',
 'pq_func_var2[40]': '1+z',
 'pq_func_par0[40]': 0.5,
 'pq_func_par1[40]': 1e10,
 'pq_func_par2[40]': 0.2,
 'pq_func_par3[40]': 5.,
 'pq_func_par4[40]': 0.0,
 'pq_val_ceil[40]': 1.0,
}

_fduty_best = \
{
 "pq_func_par0[0]": 0.0255141297829,
 "pq_func_par1[0]": 4.06615553432e+11,
 "pq_func_par3[0]": -0.791000907754,
 "pq_func_par0[40]": 0.826491313083,
 "pq_func_par2[40]": 0.492503004257,
 "pq_func_par4[40]": -0.0294978445953,
 "pq_func_par0[22]": 0.898270573333,
 "pq_func_par2[22]": 0.700416744489,
 "pq_func_par3[22]": 0.0906373682257,
 "pq_func_par1[22]": 1.0445802287e+12,
 "pq_func_par0[33]": 0.125370496993,
}

_fgrowth = \
{
 'pop_dust_growth': 'pq[60]',
 'pq_func_var[60]': 'Mh',
 'pq_func_var2[60]': '1+z',
 'pq_func[60]': 'pl_evolN',
 'pq_func_par0[60]': 3e10, # in yr
 'pq_func_par1[60]': 3e11,
 'pq_func_par2[60]': 0.,
 'pq_func_par3[60]': 5.,
 'pq_func_par4[60]': 0.0,
}

_kappa_evol = \
{
 "pq_func[20]": 'pl_evolS2',
 'pq_func_var[20]': 'wave',
 'pq_func_var2[20]': '1+z',
 'pq_func_var_lim[20]': (912., np.inf),
 'pq_func_var_fill[20]': 0.0,
 'pq_func_par0[20]': 1e5 * (1e3 / 1600.),      # opacity at wavelength below
 'pq_func_par1[20]': 1.6e3,
 'pq_func_par2[20]': -1.,
 'pq_func_par3[20]': 5.,        # time evolution
 'pq_func_par4[20]': 0.,        # time evolution
 'pq_func_par5[20]': 1e10,      # Mh-dep
 'pq_func_par6[20]': 0.,        # Mh-dep

}

ereg_eduty = univ.copy()
ereg_eduty.update(_fduty)
ereg_eduty.update(_fduty_best)
ereg_eduty.update(_evol_ereg_dplrd)

ereg_egrowth = univ.copy()
ereg_egrowth.update(_fgrowth)
ereg_egrowth.update(_evol_ereg_dplrd)

ereg_ekappa = univ.copy()
ereg_ekappa.update(_kappa_evol)
ereg_ekappa.update(_evol_ereg_dplrd)

mreg_eduty = ereg_eduty.copy()
mreg_eduty.update(_evol_mreg_dplrd)


_dtmr = \
{
 "pop_dust_yield": 'pq[50]',
 "pq_func[50]": 'pl_evolN',
 'pq_func_var[50]': 'Mh',
 'pq_func_var2[50]': '1+z',
 'pq_func_par0[50]': 0.1,
 'pq_func_par1[50]': 1e11,
 'pq_func_par2[50]': 0.,
 'pq_func_par3[50]': 5.,
 'pq_func_par4[50]': 0.,
 'pq_val_ceil[50]': 1.0,
}

_dtmr_best = \
{
 "pq_func_par0[0]": 0.0270948826563,
 "pq_func_par1[0]": 4.34144238121e+11,
 "pq_func_par3[0]": -0.929436652888,
 "pq_func_par0[22]": 1.16510135547,
 "pq_func_par2[22]": 0.672887955656,
 "pq_func_par3[22]": 0.192258001389,
 "pq_func_par1[22]": 3.80649012601e+11,
 "pq_func_par0[50]": 0.278584104596,
 "pq_func_par2[50]": -0.085758644535,
 "pq_func_par4[50]": 0.751410876772,
 "pq_func_par0[33]": 0.181775151977,
}

_duty_dtmr_best_ereg = \
{
'pq_func_par0[0]': 0.1675,
'pq_func_par1[0]': 59962271529.3613,
'pq_func_par3[0]': -0.3825,
'pq_func_par0[40]': 0.5841,
'pq_func_par2[40]': 0.1258,
'pq_func_par4[40]': -1.8654,
'pq_func_par0[22]': 0.8626,
'pq_func_par2[22]': 0.9404,
'pq_func_par1[22]': 92768180539.0610,
'pq_func_par0[50]': 0.4333,
'pq_func_par2[50]': 0.2162,
'pq_func_par4[50]': -1.3150,
'pq_func_par0[33]': 0.0154,
}

_duty_dtmr_best_mreg = \ # NEEDS UPDATING
{
'pq_func_par0[0]': 0.1675,
'pq_func_par1[0]': 59962271529.3613,
'pq_func_par3[0]': -0.3825,
'pq_func_par0[40]': 0.5841,
'pq_func_par2[40]': 0.1258,
'pq_func_par4[40]': -1.8654,
'pq_func_par0[22]': 0.8626,
'pq_func_par2[22]': 0.9404,
'pq_func_par1[22]': 92768180539.0610,
'pq_func_par0[50]': 0.4333,
'pq_func_par2[50]': 0.2162,
'pq_func_par4[50]': -1.3150,
'pq_func_par0[33]': 0.0154,
}

ereg = univ.copy()
ereg['pop_dust_yield'] = 0.05
ereg.update(_evol_ereg_dplrd)

ereg_edtmr = univ.copy()
ereg_edtmr.update(_dtmr)
ereg_edtmr.update(_dtmr_best)
ereg_edtmr.update(_evol_ereg_dplrd)

ereg_ekappa_edtmr = ereg_edtmr.copy()
ereg_ekappa_edtmr.update(_kappa_evol)

ereg_eduty_edtmr = ereg_edtmr.copy()
ereg_eduty_edtmr.update(_fduty)
ereg_eduty_edtmr.update(_duty_dtmr_best_ereg)

evol = ereg_eduty_edtmr

mreg_edtmr = ereg_edtmr.copy()
mreg_edtmr.update(_evol_mreg_dplrd)

mreg_eduty_edtmr = mreg_edtmr.copy()
mreg_eduty_edtmr.update(_fduty)
mreg_eduty_edtmr.update(_duty_dtmr_best_mreg)


univ_plRd = univ.copy()
_univ_plRd_best = \
{
 "pq_func_par0[0]": 0.06358664984423915,
 "pq_func_par1[0]": 383422238216.8667,
 "pq_func_par2[0]": 0.6176456842054336,
 "pq_func_par3[0]": -1.2836704145518527,
 "pq_func_par0[22]": 1.530198199301002,
 "pq_func_par2[22]": 0.49041401581181127,
 "pq_func_par0[33]": 0.19401544545500612,
}
univ_plRd.update(_univ_plRd_best)

ereg_epeak_plRd = ereg_epeak.copy()
ereg_eduty_plRd = ereg_eduty.copy()
ereg_edtmr_plRd = ereg_edtmr.copy()
mreg_epeak_plRd = mreg_epeak.copy()
mreg_eduty_plRd = mreg_eduty.copy()
mreg_edtmr_plRd = mreg_edtmr.copy()

univ_plRd.update(_screen)
ereg_epeak_plRd.update(_screen)
ereg_eduty_plRd.update(_screen)
ereg_edtmr_plRd.update(_screen)
mreg_epeak_plRd.update(_screen)
mreg_eduty_plRd.update(_screen)
mreg_edtmr_plRd.update(_screen)

# These parameters are designed to reproduce Park et al. (2019) src model
#smhm = \
#{
# 'pop_Tmin{0}': None,
# 'pop_Mmin{0}': 1e5,  # Let focc do the work.
# 'pop_sfr_model{0}': 'smhm-func',
#
# 'pop_tstar{0}': 0.5,
# 'pop_fstar{0}': 'pq[0]',
# 'pq_func[0]{0}': 'pl',
# 'pq_func_par0[0]{0}': 0.05,
# 'pq_func_par1[0]{0}': 1e10,
# 'pq_func_par2[0]{0}': 0.5,
# 'pq_val_ceil[0]{0}': 1.,
#
# # Need something close to kappa_UV = 1.15e-28 Msun/yr/(erg/s/Hz)
# # 2x solar metallicity with BPASS single-stars is pretty close.
# 'pop_Z{0}': 0.04,
#
# 'pop_focc{0}': 'pq[40]',
# "pq_func[40]{0}": 'exp-',
# 'pq_func_var[40]{0}': 'Mh',
# 'pq_func_par0[40]{0}': 1.,
# 'pq_func_par1[40]{0}': 5e8,
# 'pq_func_par2[40]{0}': -1.,
#
# 'pop_sfr_cross_threshold{0}': False,
#
# #'pop_ion_src_cgm{0}': False,
#
# # KLUDGE-TOWN
# 'pop_fesc_LW{0}': 1.0,
#
# 'pop_sfr_model{1}': 'link:sfrd:0',
# 'pop_rad_yield{1}': 10**40.5,
# 'pop_alpha{1}': -1.,
# 'pop_Emin{1}': 500.,
# 'pop_Emax{1}': 3e4,
# 'pop_EminNorm{1}': 500.,
# 'pop_EmaxNorm{1}': 2e3,
# 'pop_ion_src_igm{1}': 1,
#}
