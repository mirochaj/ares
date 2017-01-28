from numpy import inf

# Calibration set!
energy = \
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
 
 'pop_fstar_max{0}': 0.1,            # fstar <= this value
 
 # Solve LWB!
 'pop_solve_rte{0}': (10.2, 13.6),
 'pop_tau_Nz{1}': 1e3,

 
 # SFE (through mass loading factor)
 'pop_mlf{0}': 'pq[0]',
 'pq_func{0}[0]': 'pl',
 'pq_func_var{0}[0]': 'Mh',
 
 ##
 # Steve's Equation 13.
 ##
 'pq_func_par0{0}[0]': 10.,          # really = 10 * epsilon_K * omega_49
 'pq_func_par1{0}[0]': 10**11.5,
 'pq_func_par2{0}[0]': -2./3.,   
 'pop_calib_L1600{0}': None,
 
 # Redshift dependence
 'pq_faux{0}[0]': 'pl',
 'pq_faux_var{0}[0]': '1+z',
 'pq_faux_meth{0}[0]': 'multiply',
 'pq_faux_par0{0}[0]': 1.,
 'pq_faux_par1{0}[0]': 9.,
 'pq_faux_par2{0}[0]': -1.,  # Specific to energy!
 ##
 #
 ##

 # Massive end
 'pop_fshock{0}': 'pq[1]',
 'pq_func{0}[1]': 'pl',
 'pq_func_var{0}[1]': 'Mh',
 'pq_val_ceil{0}[1]': 1.0,           # fshock <= 1

 # Steve's Equation 6 (from Faucher-Giguere+ 2011)
 'pq_func_par0{0}[1]': 0.47,         
 'pq_func_par1{0}[1]': 1e12,
 'pq_func_par2{0}[1]': -0.25,
 
 # Redshift dependence
 'pq_faux{0}[1]': 'pl',
 'pq_faux_var{0}[1]': '1+z',
 'pq_faux_meth{0}[1]': 'multiply',
 'pq_faux_par0{0}[1]': 1.,
 'pq_faux_par1{0}[1]': 4.,
 'pq_faux_par2{0}[1]': 0.38,  # Specific to energy!
  
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

momentum = energy.copy()
momentum['pq_func_par2{0}[0]'] = -1./3.
momentum['pq_faux_par2{0}[0]'] = -0.5
