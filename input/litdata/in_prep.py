import numpy as np
from mirocha2017 import base as _base_
from mirocha2017 import dflex as _dflex_
from ares.physics.Constants import E_LyA

base = _base_.copy()

_base = \
{
 'pop_sfr_model{0}': 'ensemble', 
 'pop_sed{0}': 'eldridge2009',
 
 # SFE
 'pop_fstar{0}': 'pq[0]',
 'pq_func[0]{0}': 'dpl_evolNP',
 'pq_func_var[0]{0}': 'Mh',
 'pq_func_var2[0]{0}': '1+z',
 
 # DPL in Mh
 'pq_func_par0[0]{0}': 0.05,           # Table 1 in paper (last 4 rows)
 'pq_func_par1[0]{0}': 2.8e11,
 'pq_func_par2[0]{0}': 0.49,       
 'pq_func_par3[0]{0}': -0.61,      
 'pq_func_par4[0]{0}': 1e10,  
 'pq_func_par5[0]{0}': 5.,    # 1+z pivot
 'pq_func_par6[0]{0}': 0.0,   # norm
 'pq_func_par7[0]{0}': 0.0,   # Mp 
 
 # Spectral synthesis
 'pop_sed_degrade{0}': 10,
 'pop_thin_hist{0}': 10,
 'pop_aging{0}': True,
 'pop_ssp{0}': True,
 'pop_mass_yield{0}': 0.15,
 'pop_calib_L1600{0}': None,
 'pop_Z{0}': 0.002, 
 'pop_zdead{0}': 3.5,
 
 # Synthesis control
 'pop_synth_cache_level{0}': 0,    # 1 = more careful = slower
 'pop_synth_minimal{0}': True,
 'pop_Tmin{0}': 2e4,
 
 # Metallicity evolution!?
 'pop_enrichment{0}': False,
 'pop_metal_yield{0}': 0.1,
 'pop_mass_yield{0}': 0.15,
 'pop_fpoll{0}': 1,
 
 'pop_scatter_mar{0}': 0.3,
   
 'hmf_dt': 1.,
 'hmf_tmax': 2e3,
 'hmf_model': 'Tinker10',
 
 "sigma_8": 0.8159, 
 'primordial_index': 0.9652, 
 'omega_m_0': 0.315579, 
 'omega_b_0': 0.0491, 
 'hubble_0': 0.6726,
 'omega_l_0': 1. - 0.315579,

}

base.update(_base)

_screen = \
{
 
 'pop_dust_yield{0}': 0.4,
 
 # Dust opacity vs. wavelength    
 "pop_dust_kappa{0}": 'pq[20]',   # opacity in [cm^2 / g]
 "pq_func[20]{0}": 'pl',
 'pq_func_var[20]{0}': 'wave',
 'pq_func_par0[20]{0}': 1e5,      # opacity at wavelength below
 'pq_func_par1[20]{0}': 1e3,
 'pq_func_par2[20]{0}': -1.,
 
 # Screen parameters
 'pop_dust_fcov{0}': 1,  
 "pop_dust_scale{0}": 'pq[22]',       # Scale radius [in kpc]
 "pq_func[22]{0}": 'pl_evolN',
 'pq_func_var[22]{0}': 'Mh',
 'pq_func_var2[22]{0}': '1+z',
 'pq_func_par0[22]{0}': 1.6,     # Note that Rhalo ~ Mh^1/3 / (1+z)
 'pq_func_par1[22]{0}': 1e10,
 'pq_func_par2[22]{0}': 0.45,
 'pq_func_par3[22]{0}': 5.,
 'pq_func_par4[22]{0}': 0.,    
}

screen = _screen.copy()
#screen.update(_screen_best)

_screen_dpl = \
{
 "pq_func[22]{0}": 'dpl_evolN',
 'pq_func_par0[22]{0}': 1.6,     # Normalization of length scale
 'pq_func_par1[22]{0}': 3e11,    # normalize at Mh=1e10
 'pq_func_par2[22]{0}': 0.45,    # low-mass sope
 'pq_func_par3[22]{0}': 0.45,    # high-mass slope
 'pq_func_par4[22]{0}': 1e10,    # peak mass
 'pq_func_par5[22]{0}': 5.,      # pin to z=4
 'pq_func_par6[22]{0}': 0.0      # no z evolution by default
}

screen_dpl = _screen.copy()
screen_dpl.update(_screen_dpl)

_patchy = \
{
 'pop_dust_fcov{0}': 'pq[21]',
 'pq_func[21]{0}': 'log_tanh_abs',
 'pq_func_var[21]{0}': 'Mh',
 'pq_func_par0[21]{0}': 0.05,
 'pq_func_par1[21]{0}': 1.0,
 'pq_func_par2[21]{0}': 'pq[24]',
 'pq_func_par3[21]{0}': 0.2,
 
 # Redshift evolution of transition mass
 'pq_func[24]{0}': 'linear',
 'pq_func_var[24]{0}': '1+z',
 'pq_func_par0[24]{0}': 10.8,
 'pq_func_par1[24]{0}': 5.,
 'pq_func_par2[24]{0}': 0.0,
}

patchy = screen.copy()
patchy.update(_patchy)

fduty = \
{
 'pop_fduty{0}': 'pq[40]',
 "pq_func[40]{0}": 'pl_evolN',
 'pq_func_var[40]{0}': 'Mh',
 'pq_func_var2[40]{0}': '1+z',
 'pq_func_par0[40]{0}': 0.5,
 'pq_func_par1[40]{0}': 1e10,
 'pq_func_par2[40]{0}': 0.2,
 'pq_func_par3[40]{0}': 5.,
 'pq_func_par4[40]{0}': 0.0,
 'pq_val_ceil[40]{0}': 1.0,
}

fyield = \
{
 "pop_dust_yield{0}": 'pq[50]',
 "pq_func[50]{0}": 'pl_evolN',
 'pq_func_var[50]{0}': 'Mh',
 'pq_func_var2[50]{0}': '1+z',
 'pq_func_par0[50]{0}': 0.1,
 'pq_func_par1[50]{0}': 1e11,
 'pq_func_par2[50]{0}': 0.,
 'pq_func_par3[50]{0}': 5.,
 'pq_func_par4[50]{0}': 0.,
 'pq_val_ceil[50]{0}': 0.4,
}

quench = \
{
 'pop_fduty{0}': 'pq[40]',
 "pq_func[40]{0}": 'pl',
 'pq_func_var[40]{0}': 'Mh',
 'pq_func_par0[40]{0}': 'pq[41]',
 'pq_func_par1[40]{0}': 1e10,
 'pq_func_par2[40]{0}': 0.2,
 'pq_val_ceil[40]{0}': 1.0,
 
 # Redshift evolution of fduty normalization
 'pq_func[41]{0}': 'pl',
 'pq_func_var[41]{0}': '1+z',
 'pq_func_par0[41]{0}': 0.5,
 'pq_func_par1[41]{0}': 5.,
 'pq_func_par2[41]{0}': 0.0,
 
}

destruction = \
{
 'pq_func_par2[22]{0}': 0.,    
 'pq_func_par0[1]{0}': 0.0414924616502775,
 'pq_func_par0[2]{0}': 228025945619.58618,
 'pq_func_par0[3]{0}': 0.6314799815636458,
 'pq_func_par0[4]{0}': -0.5599463903428499,
 'pq_func_par0[23]{0}': 1.5452880256044765,
 'pq_func_par2[27]{0}': -1.4360103791063477,
 'pq_func_par4[27]{0}': 0.33198068910686596,
}

_scatter = \
{
 "pop_dust_scatter{0}": 'pq[33]',
 "pq_func[33]{0}": 'pl_evolN',
 'pq_func_var[33]{0}': 'Mh',
 'pq_func_var2[33]{0}': '1+z',
 'pq_func_par0[33]{0}': 0.3,
 'pq_func_par1[33]{0}': 1e10,
 'pq_func_par2[33]{0}': 0.,
 'pq_func_par3[33]{0}': 5.,
 'pq_func_par4[33]{0}': 0., 
}

scatter = screen.copy()
scatter.update(_scatter)

_composition = \
{
 'pq_func_par2[20]{0}': 'pq[31]',
 'pq_func[31]{0}': 'pl',
 'pq_func_var[31]{0}': 'Mh',
 'pq_func_par0[31]{0}': -1.,
 'pq_func_par1[31]{0}': 1e10,
 'pq_func_par2[31]{0}': 0.,
 
 # Best fit from scatter model
 'pq_func_par2[22]{0}': 0.,
 'pq_func_par0[23]{0}': 3.3435661993369257,
 'pq_func_par0[1]{0}': 0.03599536365742767,
 'pq_func_par0[2]{0}': 167344779121.14706,
 'pq_func_par0[3]{0}': 1.1677597636641468,
 'pq_func_par0[4]{0}': 0.06069676909919603,
}


composition = screen.copy()
composition.update(_composition)

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


