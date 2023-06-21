"""
Mirocha, Furlanetto, and Sun (2017).

Parameters defining the fiducial model (see Table 1).
"""

from numpy import inf
from ares.physics.Constants import E_LyA, E_LL

# Calibration set!
dpl = \
{

 # For reionization problem
 'load_ics': True,
 'cosmological_ics': True,
 'grid_cells': 1,

 # Halos, MAR, etc.
 'pop_sfr_model{0}': 'sfe-func',
 'pop_MAR{0}': 'hmf',
 'pop_Tmin{0}': 1e4,

 "pop_Emin{0}": 10.2,
 "pop_Emax{0}": 24.6,
 "pop_EminNorm{0}": E_LL,
 "pop_EmaxNorm{0}": 24.6,

 "pop_lya_src{0}": True,
 "pop_ion_src_cgm{0}": True,
 "pop_ion_src_igm{0}": False,
 "pop_heat_src_cgm{0}": False,
 "pop_heat_src_igm{0}": False,

 # Stellar pop + fesc
 'pop_sed{0}': 'eldridge2009',
 'pop_binaries{0}': False,
 'pop_Z{0}': 0.02,
 'pop_Emin{0}': E_LyA,
 'pop_Emax{0}': 24.6,
 'pop_rad_yield{0}': 'from_sed', # EminNorm and EmaxNorm arbitrary now
                                 # should make this automatic

 'pop_fesc{0}': 0.2,

 # Solve LWB!
 'pop_solve_rte{0}': (E_LyA, E_LL),

 # SFE
 'pop_fstar{0}': 'pq[0]',
 'pq_func[0]{0}': 'dpl_normP',
 'pq_func_var[0]{0}': 'Mh',

 ##
 # IMPORTANT
 ##
 'pq_func_par0[0]{0}': 0.05,           # Table 1 in paper (last 4 rows)
 'pq_func_par1[0]{0}': 2.8e11,
 'pq_func_par2[0]{0}': 0.49,
 'pq_func_par3[0]{0}': -0.61,
 'pop_calib_wave{0}': 1600,
 'pop_calib_lum{0}': 1.0185e28,      # Enforces Equation 13 in paper
 ##
 #
 ##

 'pop_sfr_model{1}': 'link:sfrd:0',
 'pop_Tmin{1}': 'pop_Tmin{0}',
 "pop_lya_src{1}": False,
 "pop_ion_src_cgm{1}": False,
 "pop_ion_src_igm{1}": True,
 "pop_heat_src_cgm{1}": False,
 "pop_heat_src_igm{1}": True,


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
 'photon_counting': True,
 'cgm_initial_temperature': 2e4,
 'cgm_recombination': 'B',
 'clumping_factor': 3.,
 #'smooth_derivative': 0.5,
 'final_redshift': 5,
}

base = dpl

_floor_specific = \
{
 'pq_val_floor{0}[0]': 0.005,
}

floor = _floor_specific

_steep_specific = \
{
 'pop_focc{0}': 'pq[5]',
 'pq_func[5]{0}': 'okamoto',
 'pq_func_var[5]{0}': 'Mh',
 'pq_func_par0[5]{0}': 1.,
 'pq_func_par1[5]{0}': 1e9,
}

steep = _steep_specific

"""
Redshift-dependent options.
"""

_flex = \
{
 'pq_func[0]{0}': 'dpl_evolNP',
 'pq_func_var[0]{0}': 'Mh',
 'pq_func_var2[0]{0}': '1+z',

 'pq_val_ceil[0]{0}': 1.0,

 # Standard dpl model at 10^8 Msun
 'pq_func_par0[0]{0}': 0.05,
 'pq_func_par1[0]{0}': 2.8e11,
 'pq_func_par2[0]{0}': 0.49,
 'pq_func_par3[0]{0}': -0.61,
 'pq_func_par4[0]{0}': 1e10,        # Mass at which fstar,0 is defined
 'pq_func_par5[0]{0}': 5.,

 # Evolving bits
 'pq_func_par6[0]{0}': 0.,   # power-law index!
 'pq_func_par7[0]{0}': 0.,   # power-law index!

}

flex = _flex

_flex2 = \
{
 'pq_func[0]{0}': 'dpl_evolNPSF',
 'pq_func_var[0]{0}': 'Mh',
 'pq_func_var2[0]{0}': '1+z',

 'pq_val_ceil[0]{0}': 1.0,

 # Standard dpl model at 10^8 Msun
 'pq_func_par0[0]{0}': 0.019,
 'pq_func_par1[0]{0}': 2.8e11,
 'pq_func_par2[0]{0}': 0.49,
 'pq_func_par3[0]{0}': -0.61,
 'pq_func_par4[0]{0}': 1e10,        # Mass at which fstar,0 is defined
 'pq_func_par5[0]{0}': 7.,
 'pq_func_par6[0]{0}': 0.,          # Redshift evolution of norm
 'pq_func_par7[0]{0}': 0.,          # Redshift evolution of peak
 'pq_func_par8[0]{0}': 0.,          # Redshift evolution of low-mass slope
 'pq_func_par9[0]{0}': 0.,          # Redshift evolution of high-mass slope

 # Floor parameters
 'pq_func_par10[0]{0}': 1e-6,
 'pq_func_par11[0]{0}': 0.0,

 # Okamoto parameters
 'pop_focc{0}': 'pq[1]',
 'pq_func[1]{0}': 'okamoto_evol',
 'pq_func_var[1]{0}': 'Mh',
 'pq_func_var2[1]{0}': '1+z',
 'pq_func_par0[1]{0}': 0.0,   # exponent
 'pq_func_par1[1]{0}': 0.0,   # critical mass
 'pq_func_par2[1]{0}': 7.0,   # pivot redshift
 'pq_func_par3[1]{0}': 0.0,
 'pq_func_par4[1]{0}': 0.0,
}

dflex = _flex2
