import ares
import numpy as np
from mirocha2016 import dpl

# relative to mirocha2016:dpl
_generic_updates = \
{
 'initial_redshift': 60,
 'pop_zform{0}': 60,
 'pop_zform{1}': 60,
 'feedback_LW': True,
}

# This can lead to pickling issues...argh
halos = ares.physics.HaloMassFunction()

barrier_M = lambda zz: halos.VirialMass(300, zz)
barrier_A = lambda zz: halos.VirialMass(1e4, zz)
    
log_barrier_M = lambda zz: np.log10(barrier_M(zz))    
log_barrier_A = lambda zz: np.log10(barrier_A(zz))

_csfr = \
{

 'pop_zform{0}': 60,
 'pop_zform{1}': 60,
 'pop_zform{2}': 60,
 'pop_zform{3}': 60,
 
 'pop_Tmin_ceil{0}': 1e4,

 'pop_sfr_model{2}': 'sfr-func',
 'pop_sfr{2}': 1e-6,
 'pop_Tmax{2}': 1e4,
 'pop_sfr_cross_threshold{2}': False,
 'pop_sed{2}': 'eldridge2009',
 'pop_binaries{2}': False,
 'pop_Z{2}': 1e-3,
 'pop_Emin{2}': 10.19,
 'pop_Emax{2}': 24.6,
 'pop_rad_yield{2}': 'from_sed', # EminNorm and EmaxNorm arbitrary now
                                  # should make this automatic
 
 'pop_heat_src_igm{2}': False,
 'pop_ion_src_igm{2}': False,
 
  # Solve LWB!
 'pop_solve_rte{2}': (10.2, 13.6),
 
 # Radiative knobs
 'pop_fesc_LW{2}': 1.,
 'pop_fesc{2}': 0.0,
 'pop_rad_yield{3}': 2.6e39,
 
 # Other stuff needed for X-rays
 'pop_sfr_model{3}': 'link:sfrd:2',
 'pop_sed{3}': 'mcd',
 'pop_Z{3}': 1e-3,
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
 
 'pop_Tmin{0}': 1e4,
 #'pop_Tmin_ceil{0}': 1e4,
 #'pop_Mmin{0}': 'link:Mmax_active:2',
 
 # Tmin here just an initial guess -- will get modified by feedback.
 'pop_Tmin{2}': 500.,
 'pop_Mmin{3}': 'pop_Mmin{2}',
 'pop_Tmin{3}': None,
 'pop_Tmax{2}': 1e4,

}


csfr = dpl.copy()
csfr.update(_generic_updates)
csfr.update(_csfr)

xsfe = dpl.copy()

_step_specific = \
{
 
 'pop_fesc_LW{0}': 'pq[101]',
 'pq_func{0}[101]': 'astep',
 'pq_func_var{0}[101]': 'Mh',
 'pq_func_par0{0}[101]': 1.,
 'pq_func_par1{0}[101]': 1.,
 'pq_func_par2{0}[101]': (barrier_A, 'z', 1),
 
 'pop_fesc{0}': 'pq[102]',
 'pq_func{0}[102]': 'astep',
 'pq_func_var{0}[102]': 'Mh',
 'pq_func_par0{0}[102]': 0., # No LyC from minihalos by default
 'pq_func_par1{0}[102]': 0.1,
 'pq_func_par2{0}[102]': (barrier_A, 'z', 1),

 'pop_Tmin{0}': 500.,

 # X-ray sources
 'pop_rad_yield{1}': 'pq[103]',
 'pq_func{1}[103]': 'astep',
 'pq_func_var{1}[103]': 'Mh',
 'pq_func_par0{1}[103]': 2.6e39,
 'pq_func_par1{1}[103]': 2.6e39,
 'pq_func_par2{1}[103]': (barrier_A, 'z', 1),
}

step = dpl.copy()
step.update(_step_specific)
step.update(_generic_updates)

_exp_specific = \
{
  'initial_redshift': 60,
  'pop_zform{0}': 60,
  'pop_zform{1}': 60,
  'pop_zform{2}': 60,
  'pop_zform{3}': 60,
  
  'pop_sfr_model{2}': 'sfe-func',
  'pop_fstar{2}': 'pq[101]',
  'pop_fstar_negligible{2}': 1e-5, 
  
  'pq_func{2}[101]': 'exp',
  'pq_func_var{2}[101]': 'Mh',
  'pq_func_par0{2}[101]': 1e-2,
  
  ##
  # Might want to change this
  ##
  'pq_func_par1{2}[101]': ('pop_Mmin', 'z', 1),
  'pq_func_par2{2}[101]': 1.,
  
  'pop_sed{2}': 'eldridge2009',
  'pop_binaries{2}': False,
  'pop_Z{2}': 1e-3,
  'pop_Emin{2}': 10.19,
  'pop_Emax{2}': 24.6,
  'pop_rad_yield{2}': 'from_sed', # EminNorm and EmaxNorm arbitrary now
                                   # should make this automatic
  
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
  'pop_Z{3}': 1e-3,
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
  'pop_Mmin{3}': 'pop_Mmin{2}',
  'pop_Tmin{3}': None,
  'pop_Tmax{2}': 1e4,

  # Feedback
  'feedback_LW': True,
  'feedback_LW_Mmin': 'visbal2015',
  'feedback_LW_Tcut': 1e4, 
}

exp = dpl.copy()
exp.update(_exp_specific)

exp_blobs = \
{
 'blob_names': ['popII_sfrd_tot', 'popIII_sfrd_tot', 
                'popII_sfrd_bc',  'popIII_sfrd_bc', 
                'popII_Mmin', 'popIII_Mmin'],
 'blob_ivars': ('z', np.arange(5, 60.1, 0.1)),
 'blob_funcs': ['pops[0].SFRD', 'pops[2].SFRD', 'pops[0].SFRD_at_threshold',
    'pops[2].SFRD_at_threshold', 'pops[0].Mmin', 'pops[2].Mmin'],
}

# This is a little trickier
step_blobs = \
{
 'blob_names': ['popII_sfrd_tot', #'popIII_sfrd_tot', 
                'popII_sfrd_bc',  #'popIII_sfrd_bc', 
                'popII_Mmin'],
 'blob_ivars': ('z', np.arange(5, 60.1, 0.1)),
 'blob_funcs': ['pops[0].SFRD', 'pops[0].SFRD_at_threshold', 'pops[0].Mmin'],
}




