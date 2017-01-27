"""
Mirocha, Sun, and Furlanetto (2016), in prep.

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

 'pop_fesc{0}': 0.1,
 
 # Solve LWB!
 'pop_solve_rte{0}': (10.2, 13.6),

 
 # SFE
 'pop_fstar{0}': 'pq[0]',
 'pq_func{0}[0]': 'dpl',
 'pq_func_var{0}[0]': 'Mh',
 
 ##
 # IMPORTANT
 ##
 'pq_func_par0{0}[0]': 0.05,       # Table 1 in paper (last 4 rows)
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
 'pop_alpha{1}': -1.5,
 'pop_Emin{1}': 2e2,
 'pop_Emax{1}': 3e4,
 'pop_EminNorm{1}': 5e2,
 'pop_EmaxNorm{1}': 8e3,
 'pop_logN{1}': -inf,

 'pop_solve_rte{1}': True,
 'pop_tau_Nz{1}': 1e3,
 'pop_approx_tau{1}': 'neutral',

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
'pq_faux{0}[0]': 'plexp',
'pq_faux_var{0}[0]': 'Mh',
'pq_faux_meth{0}[0]': 'add',
'pq_faux_par0{0}[0]': 0.005,
'pq_faux_par1{0}[0]': 1e9,
'pq_faux_par2{0}[0]': 0.01,
'pq_faux_par3{0}[0]': 1e10,
'pq_faux_par4{0}[0]': 1.,
}

floor = dpl.copy()
floor.update(_floor_specific)

_steep_specific = \
{
 'pq_faux{0}[0]': 'okamoto',
 'pq_faux_var{0}[0]': 'Mh',
 'pq_faux_meth{0}[0]': 'multiply',
 'pq_faux_par0{0}[0]': 1.,
 'pq_faux_par1{0}[0]': 1e9,
}

steep = dpl.copy()
steep.update(_steep_specific)

"""
Redshift-dependent options.
"""
_fz_specific = \
{
 'pq_faux{0}[0]': 'pl',
 'pq_faux_var{0}[0]': '1+z',
 'pq_faux_meth{0}[0]': 'multiply',
 'pq_faux_par0{0}[0]': 1.,
 'pq_faux_par1{0}[0]': 7.,
 'pq_faux_par2{0}[0]': 0.,
}

_Mz_specific = \
{
 'pq_func_par1{0}[0]': 'pl',
 'pq_func_par1_par0{0}[0]': dpl['pq_func_par1{0}[0]'],
 'pq_func_par1_par1{0}[0]': 5.9,
 'pq_func_par1_par2{0}[0]': -1.,
}

_flex = \
{
 'pq_func{0}[0]': 'dpl_arbnorm',
 'pq_func_var{0}[0]': 'Mh',
 
 # Standard dpl model at 10^8 Msun
 'pq_func_par0{0}[0]': 0.00205,

 'pq_func_par2{0}[0]': 0.49,       
 'pq_func_par3{0}[0]': -0.61,      
 'pq_func_par4{0}[0]': 1e8,        # Mass at which fstar,0 is defined
 
 # Redshift evolution peak mass only
 'pq_func_par1{0}[0]': 'pl',
 'pq_func_par1_par0{0}[0]': 2.8e11,
 'pq_func_par1_par1{0}[0]': 7.,
 'pq_func_par1_par2{0}[0]': -1.,
 
 'pop_calib_L1600{0}': 1.0185e28,
 
 # Redshift evolution in normalization of SFE
 'pq_faux{0}[0]': 'pl',
 'pq_faux_var{0}[0]': '1+z',
 'pq_faux_meth{0}[0]': 'multiply',
 'pq_faux_par0{0}[0]': 1.,
 'pq_faux_par1{0}[0]': 7.,
 'pq_faux_par2{0}[0]': -1.,

}

dpl_fz = dpl.copy()
dpl_fz.update(_fz_specific)
dpl_Mz = dpl.copy()
dpl_Mz.update(_Mz_specific)
dpl_flex = dpl.copy()
dpl_flex.update(_flex)

_steep_fz = {}
for key in _fz_specific:
    new_key = 'pq_faux%s' % key.split('faux')[1]
    _steep_fz[new_key] = _fz_specific[key]

for key in _steep_specific:
    new_key = 'pq_faux_A%s' % key.split('faux')[1]
    _steep_fz[new_key] = _steep_specific[key]

steep_fz = dpl.copy()
steep_fz.update(_steep_fz)

_floor_fz = {}

for key in _fz_specific:
    new_key = 'pq_faux%s' % key.split('faux')[1]
    _floor_fz[new_key] = _fz_specific[key]

for key in _floor_specific:
    new_key = 'pq_faux_A%s' % key.split('faux')[1]
    _floor_fz[new_key] = _floor_specific[key]

floor_fz = dpl.copy()
floor_fz.update(_floor_fz)

