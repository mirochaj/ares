"""
Mirocha, Furlanetto, and Sun (2017).

Parameters defining the fiducial model (see Table 1).
"""

from numpy import inf
from ares.physics.Constants import E_LyA

# Calibration set!
dpl = \
{

 # Halos, MAR, etc.
 'pop_Tmin{0}': 1e4,
 'pop_Tmin{1}': 'pop_Tmin{0}',
 'pop_sfr_model{0}': 'sfe-func',
 'pop_sfr_model{1}': 'link:sfrd:0',
 'pop_MAR{0}': 'hmf',
 'pop_MAR_conserve_norm{0}': False,
 
 # Stellar pop + fesc
 'pop_sed{0}': 'eldridge2009',
 'pop_binaries{0}': False,
 'pop_Z{0}': 0.02,
 'pop_Emin{0}': 10.19,
 'pop_Emax{0}': 24.6,
 'pop_rad_yield{0}': 'from_sed', # EminNorm and EmaxNorm arbitrary now
                                 # should make this automatic

 'pop_fesc{0}': 0.2,
 
 # Solve LWB!
 'pop_solve_rte{0}': (10.2, 13.6),
 
 # SFE
 'pop_fstar{0}': 'pq[0]',
 'pq_func{0}[0]': 'dpl',
 'pq_func_var{0}[0]': 'Mh',
 
 ##
 # IMPORTANT
 ##
 'pq_func_par0{0}[0]': 0.05,           # Table 1 in paper (last 4 rows)
 'pq_func_par1{0}[0]': 2.8e11,
 'pq_func_par2{0}[0]': 0.49,       
 'pq_func_par3{0}[0]': -0.61,      
 'pop_calib_L1600{0}': 1.0185e28,      # Enforces Equation 13 in paper 
 ##
 #
 ##

 # Careful with X-ray heating
 'pop_sed{1}': 'mcd',
 'pop_Z{1}': 'pop_Z{0}',
 'pop_rad_yield{1}': 2.6e39,
 'pop_rad_yield_Z_index{1}': None,
 'pop_alpha{1}': -1.5, # not used unless fesc > 0
 'pop_Emin{1}': 2e2,
 'pop_Emax{1}': 3e4,
 'pop_EminNorm{1}': 5e2,
 'pop_EmaxNorm{1}': 8e3,
 'pop_logN{1}': -inf,

 'pop_solve_rte{1}': True,
 'tau_redshift_bins': 1000,
 'tau_approx': 'neutral',

 # Control parameters
 'include_He': True,
 'approx_He': True,
 'secondary_ionization': 3,
 'approx_Salpha': 3,
 'problem_type': 102,
 'photon_counting': True,
 'cgm_initial_temperature': 2e4,
 'cgm_recombination': 'B',
 'clumping_factor': 3.,
 #'smooth_derivative': 0.5,
 'final_redshift': 5.,
}

_floor_specific = \
{
 'pq_val_floor{0}[0]': 0.005,
}

floor = _floor_specific

_steep_specific = \
{
 'pop_focc{0}': 'pq[5]',
 'pq_func{0}[5]': 'okamoto',
 'pq_func_var{0}[5]': 'Mh',
 'pq_func_par0{0}[5]': 1.,
 'pq_func_par1{0}[5]': 1e9,
}

steep = _steep_specific

"""
Redshift-dependent options.
"""

_flex = \
{
 'pq_func{0}[0]': 'dpl_arbnorm',
 'pq_func_var{0}[0]': 'Mh',

 'pq_val_ceil{0}[0]': 1.0,

 # Standard dpl model at 10^8 Msun
 'pq_func_par0{0}[0]': 'pq[1]',
 'pq_func_par1{0}[0]': 'pq[2]',
 'pq_func_par2{0}[0]': 0.49,
 'pq_func_par3{0}[0]': -0.61,
 'pq_func_par4{0}[0]': 1e8,        # Mass at which fstar,0 is defined

 # Evolving part
 'pq_func{0}[1]': 'pl',
 'pq_func_var{0}[1]': '1+z',
 'pq_func_par0{0}[1]': 0.00205,
 'pq_func_par1{0}[1]': 7.,
 'pq_func_par2{0}[1]': 0.,   # power-law index!

 'pq_func{0}[2]': 'pl',
 'pq_func_var{0}[2]': '1+z',
 'pq_func_par0{0}[2]': 2.8e11,
 'pq_func_par1{0}[2]': 7.,
 'pq_func_par2{0}[2]': 0.,   # power-law index!

}

flex = _flex

_flex2 = \
{
 'pq_func{0}[0]': 'dpl_arbnorm',
 'pq_func_var{0}[0]': 'Mh',

 'pq_val_ceil{0}[0]': 1.0,

 # Standard dpl model at 10^8 Msun
 'pq_func_par0{0}[0]': 'pq[1]',
 'pq_func_par1{0}[0]': 'pq[2]',
 'pq_func_par2{0}[0]': 'pq[3]',
 'pq_func_par3{0}[0]': 'pq[4]',
 'pq_func_par4{0}[0]': 1e10,        # Mass at which fstar,0 is defined

 # Evolving part
 'pq_func{0}[1]': 'pl',
 'pq_func_var{0}[1]': '1+z',
 'pq_func_par0{0}[1]': 0.019,       # DPL model at Mh=1e10
 'pq_func_par1{0}[1]': 7.,
 'pq_func_par2{0}[1]': 0.,   # power-law index!

 'pq_func{0}[2]': 'pl',
 'pq_func_var{0}[2]': '1+z',
 'pq_func_par0{0}[2]': 2.8e11,
 'pq_func_par1{0}[2]': 7.,
 'pq_func_par2{0}[2]': 0.,   # power-law index!

 'pq_func{0}[3]': 'pl',
 'pq_func_var{0}[3]': '1+z',
 'pq_func_par0{0}[3]': 0.49,
 'pq_func_par1{0}[3]': 7.,
 'pq_func_par2{0}[3]': 0.,   # power-law index!
 
 'pq_func{0}[4]': 'pl',
 'pq_func_var{0}[4]': '1+z',
 'pq_func_par0{0}[4]': -0.61,
 'pq_func_par1{0}[4]': 7.,
 'pq_func_par2{0}[4]': 0.,   # power-law index!
 
 # Possibility of LF steepening.
 'pq_val_floor{0}[0]': 'pq[5]',
 'pq_func{0}[5]': 'pl',
 'pq_func_var{0}[5]': '1+z',
 'pq_func_par0{0}[5]': 0.0, # unused by default
 'pq_func_par1{0}[5]': 7.,
 'pq_func_par2{0}[5]': 0.,
 
 # Possibility of LF turn-over
 'pop_focc{0}': 'pq[6]',
 'pq_func{0}[6]': 'okamoto',
 'pq_func_var{0}[6]': 'Mh',
 'pq_func_par0{0}[6]': 'pq[7]',
 'pq_func_par1{0}[6]': 'pq[8]',

 'pq_func{0}[7]': 'pl',
 'pq_func_var{0}[7]': '1+z',
 'pq_func_par0{0}[7]': 1.,
 'pq_func_par1{0}[7]': 5.,   # effectively not in use
 'pq_func_par2{0}[7]': 0.,   # power-law index!

 'pq_func{0}[8]': 'pl',
 'pq_func_var{0}[8]': '1+z',
 'pq_func_par0{0}[8]': 0.,  # Renders focc = 1 for all halos 
 'pq_func_par1{0}[8]': 7.,
 'pq_func_par2{0}[8]': 0.,   # power-law index!
}

dflex = _flex2

fobsc = \
{
 'pop_fobsc{0}': 'pq[10]',
 'pop_fobsc_by_num{0}': False,     # fraction of UV luminosity that gets out
 'pq_val_ceil{0}[10]': 1.0,
 'pq_val_floor{0}[10]': 0.0, 
 'pq_func{0}[10]': 'log_tanh_abs',
 'pq_func_var{0}[10]': 'Mh',
 'pq_func_par0{0}[10]': 0.0,       # minimal obscuration
 'pq_func_par1{0}[10]': 'pq[11]',  # peak obscuration
 'pq_func_par2{0}[10]': 'pq[12]',  # log transition mass
 'pq_func_par3{0}[10]': 'pq[13]',  # dlogM
 
 'pq_func{0}[11]': 'pl',
 'pq_func_var{0}[11]': '1+z',
 'pq_func_par0{0}[11]': 0.5,
 'pq_func_par1{0}[11]': 7.,   # effectively not in use
 'pq_func_par2{0}[11]': 0.,   # power-law index!
 
 'pq_func{0}[12]': 'pl',
 'pq_func_var{0}[12]': '1+z',
 'pq_func_par0{0}[12]': 11.,  
 'pq_func_par1{0}[12]': 7.,
 'pq_func_par2{0}[12]': 0.,   # power-law index!
 
 'pq_func{0}[13]': 'pl',
 'pq_func_var{0}[13]': '1+z',
 'pq_func_par0{0}[13]': 1.0,  
 'pq_func_par1{0}[13]': 7.,
 'pq_func_par2{0}[13]': 0.,   # power-law index!
 
}



