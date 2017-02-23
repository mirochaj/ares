import ares
import numpy as np

halos = ares.physics.HaloMassFunction()

barrier_M = lambda zz: halos.VirialMass(300, zz)
barrier_A = lambda zz: halos.VirialMass(1e4, zz)
    
log_barrier_M = lambda zz: np.log10(barrier_M(zz))    
log_barrier_A = lambda zz: np.log10(barrier_A(zz))

step = \
{
 'initial_redshift': 60,
 'pop_zform{0}': 60,
 'pop_zform{1}': 60,
 
 'pop_fesc_LW{0}': 'pq[1]',
 'pq_func{0}[1]': 'astep',
 'pq_func_var{0}[1]': 'Mh',
 'pq_func_par0{0}[1]': 1.,
 'pq_func_par1{0}[1]': 1.,
 'pq_func_par2{0}[1]': (barrier_A, 'z', 1),
 
 'pop_fesc{0}': 'pq[2]',
 'pq_func{0}[2]': 'astep',
 'pq_func_var{0}[2]': 'Mh',
 'pq_func_par0{0}[2]': 0.1,
 'pq_func_par1{0}[2]': 0.1,
 'pq_func_par2{0}[2]': (barrier_A, 'z', 1),

 'pop_Tmin{0}': 300.,

 # X-ray sources
 'pop_sfr_model{1}': 'link:sfe:0',
 'pop_Tmin{1}': 'pop_Tmin{0}',

 'pop_rad_yield{1}': 'pq[3]',
 'pq_func{1}[3]': 'astep',
 'pq_func_var{1}[3]': 'Mh',
 'pq_func_par0{1}[3]': 2.6e39,
 'pq_func_par1{1}[3]': 2.6e39,
 'pq_func_par2{1}[3]': (barrier_A, 'z', 1),

 'feedback_LW_Mmin': 'visbal2015',
 'feedback_LW_felt_by': [0,1],
 'feedback_LW_Tcut': 1e4,
 'feedback_maxiter': 15,
 'feedback_LW_rtol': 0,
 'feedback_LW_atol': 1.,
 'feedback_LW_mean_err': False,
 'feedback_LW_Mmin_uponly': False,
 'feedback_LW_Mmin_smooth': False,
}

exp_Mtr_rel = \
{
  'initial_redshift': 60,
  'pop_zform{0}': 60,
  'pop_zform{1}': 60,
  'pop_zform{2}': 60,
  'pop_zform{3}': 60,
  
  'pop_sfr_model{2}': 'sfe-func',
  'pop_fstar{2}': 'pq[1]',
  'pop_fstar_negligible{2}': 1e-5, 
  
  'pq_func{2}[1]': 'exp',
  'pq_func_var{2}[1]': 'Mh',
  
  'pq_func_par0{2}[1]': 1e-2,
  
  ##
  # Might want to change this
  ##
  'pq_func_par1{2}[1]': ('pop_Mmin', 'z', 1),
  'pq_func_par2{2}[1]': 1.,
  
  'pop_sed{2}': 'eldridge2009',
  'pop_binaries{2}': False,
  'pop_Z{2}': 0.02,
  'pop_Emin{2}': 10.19,
  'pop_Emax{2}': 24.6,
  'pop_rad_yield{2}': 'from_sed', # EminNorm and EmaxNorm arbitrary now
                                   # should make this automatic
  
  'pop_fesc{2}': 0.1,
  'pop_heat_src_igm{2}': False,
  'pop_ion_src_igm{2}': False,

   # Solve LWB!
  'pop_solve_rte{2}': (10.2, 13.6),

  # Radiative knobs
  'pop_fesc_LW{2}': 1.,
  'pop_fesc{2}': 0.0,
  'pop_rad_yield{3}': 1. * 2.6e39,

  # Other stuff needed for X-rays
  'pop_sfr_model{3}': 'link:sfrd:2',
  'pop_sed{3}': 'mcd',
  'pop_Z{3}': 'pop_Z{0}',
  'pop_rad_yield_Z_index{3}': None,
  'pop_Emin{3}': 2e2,
  'pop_Emax{3}': 3e4,
  'pop_EminNorm{3}': 5e2,
  'pop_EmaxNorm{3}': 8e3,
  'pop_ion_src_cgm{3}': False,
  'pop_logN{3}': -np.inf,

  'pop_solve_rte{3}': True,
  'pop_tau_Nz{3}': 1e3,
  'pop_approx_tau{3}': 'neutral',

  ##
  # THIS PART IS NEW AND IMPORTANT
  ##
  'pop_Tmin{0}': None,
  'pop_Tmin_ceil{0}': 1e4,
  'pop_Mmin{0}': 'link:Mmax_active:2',
  'pop_Tmin{2}': 500.,
  'pop_Tmax{2}': 1e4,

  # Feedback
  'feedback_LW_maxiter': 15,
  'feedback_LW_mean_err': False, 
  'feedback_LW_Mmin_rtol': 0,
  'feedback_LW_Mmin_atol': 1.,

  'feedback_LW_Mmin': 'visbal2015',
  'feedback_LW_felt_by': [2,3],
  'feedback_LW_Tcut': 1e4,
  'feedback_LW_Mmin_uponly': False,
  'feedback_LW_Mmin_smooth': False,
}

exp_Mtr_fix = exp_Mtr_rel.copy()
exp_Mtr_fix['pq_func_par1{2}[1]'] = 1e7

exp = exp_Mtr_rel

sfrd_blobs = \
{
 'blob_names': ['sfrd{0}', 'sfrd{2}', 'sfrd_bc{0}', 'sfrd_bc{2}', 
    'Mmin{0}', 'Mmin{2}'],
 'blob_ivars': ('z', np.arange(5, 60.1, 0.1)),
 'blob_funcs': ['pops[0].SFRD', 'pops[2].SFRD', 'pops[0].SFRD_at_threshold',
    'pops[2].SFRD_at_threshold', 'pops[0].Mmin', 'pops[2].Mmin'],
}


