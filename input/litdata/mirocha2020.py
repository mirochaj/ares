import numpy as np
from ares.physics.Constants import E_LyA, E_LL

_base = \
{
 'pop_sfr_model': 'ensemble', 
 
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
 'pop_calib_L1600': None,
 'pop_Z': 0.002, 
 'pop_zdead': 3.5,
 
 # Synthesis control
 'pop_synth_cache_level': 0,    # 1 = more careful = slower
 'pop_synth_minimal': True,
 'pop_Tmin': 2e4,
 
 # Metallicity evolution!?
 'pop_enrichment': False,
 'pop_metal_yield': 0.1,
 'pop_mass_yield': 0.15,
 'pop_fpoll': 1,
   
 # Use constant timestep
 'hmf_dt': 1.,
 'hmf_tmax': 2e3,
 'hmf_model': 'Tinker10',
 
 # Add scatter to SFRs
 'pop_scatter_mar': 0.3,
 
 # For reproducibility. 
 'pop_scatter_mar_seed': 10620202,
 'pop_dust_scatter_seed': 87112948,
 'pop_fduty_seed': 982323505,
 
 # Use cosmology consistent with Paul's simulations.
 "sigma_8": 0.8159, 
 'primordial_index': 0.9652, 
 'omega_m_0': 0.315579, 
 'omega_b_0': 0.0491, 
 'hubble_0': 0.6726,
 'omega_l_0': 1. - 0.315579,
 
}

_legacy = _base.copy()
_legacy['pop_sfr_model'] = 'sfe-func'
_legacy['pop_dust_yield'] = None
_legacy['pop_scatter_mar'] = 0
_legacy['pop_sed_degrade'] = None
_legacy['pop_thin_hist'] = 1
_legacy['pop_aging'] = False
_legacy['pop_ssp'] = False

# NEEDS UPDATE
_legacy_best = \
{
 'pq_func_par0[0]': 0.05,
 'pq_func_par1[0]': 2.8e11,
 'pq_func_par2[0]': 0.49,  
 'pq_func_par3[0]': -0.61,   
}

legacy = _legacy.copy()
legacy.update(_legacy_best)

legacy_irxb = legacy.copy()
legacy_irxb['dustcorr_method'] = 'meurer1999'
legacy_irxb['dustcorr_beta'] = 'bouwens2014'

_screen = \
{
 'pop_dust_yield': 0.4,
 
 # Dust opacity vs. wavelength    
 "pop_dust_kappa": 'pq[20]',   # opacity in [cm^2 / g]
 "pq_func[20]": 'pl',
 'pq_func_var[20]': 'wave',
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
_evol = \
{
'pq_func_par2[0]': 0.666,
'pq_func_par6[0]': 1.0,
'pq_func_par2[22]': 0.33,
'pq_func_par4[33]': -1.,
}

# Add models for no dust and IRX-beta approaches
univ = plrd.copy()
univ.update(_screen_dpl)

_univ_best = \
{
 'pq_func_par0[0]': 0.0303,
 'pq_func_par1[0]': 289049667789.4576,
 'pq_func_par2[0]': 0.6627,
 'pq_func_par3[0]': -0.5206,
 'pq_func_par0[22]': 0.9626,
 'pq_func_par2[22]': 0.6884,
 'pq_func_par3[22]': 0.1365,
 'pq_func_par1[22]': 735886684008.5193,
 'pq_func_par0[33]': 0.0856,    
}

univ.update(_univ_best)

# Not yet implemented
_univ_nodust_best = {}

univ_nodust = univ.copy()
univ_nodust['pop_dust_yield'] = None
univ_nodust.update(_univ_nodust_best)

_peak_best = \
{
 'pq_func_par0[0]': 0.0285,
 'pq_func_par1[0]': 183781388072.8878,
 'pq_func_par7[0]': -4.9649,
 'pq_func_par3[0]': 0.0286,
 'pq_func_par9[0]': -1.1267,
 'pq_func_par0[22]': 1.5598,
 'pq_func_par3[22]': 0.5158,
 'pq_func_par1[22]': 38397972076.8372,
 'pq_func_par0[33]': 0.0503,
}

evo_peak = univ.copy()
evo_peak['pq_func[0]'] = 'dpl_evolNPS'
evo_peak.update(_peak_best)
evo_peak.update(_evol)

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
 'pq_func_par0[0]': 0.0236,
 'pq_func_par1[0]': 413760555599.5621,
 'pq_func_par3[0]': -0.9839,
 'pq_func_par0[40]': 0.9067,
 'pq_func_par2[40]': 0.0133,
 'pq_func_par4[40]': -0.4039,
 'pq_func_par0[22]': 1.0544,
 'pq_func_par3[22]': 0.6609,
 'pq_func_par1[22]': 516887256852.3492,
 'pq_func_par0[33]': 0.0896,    
}

evo_duty = univ.copy()
evo_duty.update(_fduty)
evo_duty.update(_fduty_best)
evo_duty.update(_evol)

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
 'pq_func_par0[0]': 0.0224,
 'pq_func_par1[0]': 577933312045.8309,
 'pq_func_par3[0]': -0.8118,
 'pq_func_par0[22]': 1.9526,
 'pq_func_par3[22]': 0.6356,
 'pq_func_par1[22]': 103593129846.6359,
 'pq_func_par0[50]': 0.5278,
 'pq_func_par2[50]': 0.1777,
 'pq_func_par4[50]': 1.4848,
 'pq_func_par0[33]': 0.2331,    
}

evo_dtmr = univ.copy()
evo_dtmr.update(_dtmr)
evo_dtmr.update(_dtmr_best)
evo_dtmr.update(_evol)

# These parameters are designed to reproduce Park et al. (2019) src model
smhm = \
{
 'pop_Tmin{0}': None,    
 'pop_Mmin{0}': 1e5,  # Let focc do the work.
 'pop_sfr_model{0}': 'smhm-func',
 
 'pop_tstar{0}': 0.5,
 'pop_fstar{0}': 'pq[0]',
 'pq_func[0]{0}': 'pl',
 'pq_func_par0[0]{0}': 0.05,
 'pq_func_par1[0]{0}': 1e10,
 'pq_func_par2[0]{0}': 0.5,
 'pq_val_ceil[0]{0}': 1.,
    
 # Need something close to kappa_UV = 1.15e-28 Msun/yr/(erg/s/Hz)
 # 2x solar metallicity with BPASS single-stars is pretty close.
 'pop_Z{0}': 0.04,
 
 'pop_focc{0}': 'pq[40]',
 "pq_func[40]{0}": 'exp-',
 'pq_func_var[40]{0}': 'Mh',
 'pq_func_par0[40]{0}': 1.,
 'pq_func_par1[40]{0}': 5e8,
 'pq_func_par2[40]{0}': -1.,
   
 'pop_sfr_cross_threshold{0}': False,
 
 #'pop_ion_src_cgm{0}': False,
 
 # KLUDGE-TOWN
 'pop_fesc_LW{0}': 1.0,
  
 'pop_sfr_model{1}': 'link:sfrd:0',
 'pop_rad_yield{1}': 10**40.5,
 'pop_alpha{1}': -1.,
 'pop_Emin{1}': 500.,
 'pop_Emax{1}': 3e4,
 'pop_EminNorm{1}': 500.,
 'pop_EmaxNorm{1}': 2e3,
 'pop_ion_src_igm{1}': 1, 
}


