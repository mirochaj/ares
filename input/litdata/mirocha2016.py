"""
Mirocha, Sun, and Furlanetto (2016), in prep.

Parameters defining the fiducial model (see Table 1).
"""

from numpy import inf

# Calibration set!
dpl = \
{
 # Halos, MAR, etc.
 'pop_Tmin{0}': 1e4,
 'pop_Tmin{1}': 'pop_Tmin{0}',
 'pop_model{0}': 'sfe',
 'pop_model{1}': 'sfe',
 'pop_MAR{0}': 'hmf',
 'pop_MAR_conserve_norm{0}': False,
 
 # Stellar pop + fesc
 'pop_sed{0}': 'eldridge2009',
 'pop_binaries{0}': False,
 'pop_fesc{0}': 0.2,
 'pop_Z{0}': 0.02,

 # SFE
 'pop_fstar{0}': 'php[0]',
 'php_func{0}[0]': 'dpl',
 'php_func_var{0}[0]': 'mass',
 'php_func_par0{0}[0]': 0.05,
 'php_func_par1{0}[0]': 2e11,
 'php_func_par2{0}[0]': 0.66,
 'php_func_par3{0}[0]': 0.33,
 
 # Careful with X-ray heating
 'pop_sed{1}': 'mcd',
 'pop_yield{1}': 2.6e39,
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
 'problem_type': 101.2,
 'photon_counting': True,
 'cgm_initial_temperature': 2e4,
 'cgm_recombination': 'B',
 'clumping_factor': 3.,
 'smooth_derivative': 0.5,
 'final_redshift': 5.,
}


_floor_specific = \
{
'php_faux{0}[0]': 'plexp',
'php_faux_var{0}[0]': 'mass',
'php_faux_meth{0}[0]': 'add',
'php_faux_par0{0}[0]': 0.005,
'php_faux_par1{0}[0]': 1e9,
'php_faux_par2{0}[0]': 0.01,
'php_faux_par3{0}[0]': 1e10,
}

floor = dpl.copy()
floor.update(_floor_specific)

_steep_specific = \
{
 'php_faux{0}[0]': 'okamoto',
 'php_faux_par0{0}[0]': 1.,
 'php_faux_par1{0}[0]': 5e9,
}

steep = dpl.copy()
steep.update(_steep_specific)

