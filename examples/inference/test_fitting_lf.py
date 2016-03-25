"""

test_fitting_lf.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Nov 16 14:59:43 PST 2015

Description: 

"""

import ares, time
import numpy as np

#
## INPUT
blob_n1 = ['galaxy_lf']
blob_n2 = ['fstar']
blob_n3 = ['cgm_h_2', 'igm_Ts', 'igm_Tk', 'igm_dTb']
blob_n4 = ['tau_e', 'z_C', 'igm_dTb_C', 'z_D', 'igm_dTb_D']
blob_i1 = [np.array([3.8, 4.9, 5.9, 6.9, 7, 7.9, 8, 9, 10, 10.4, 11, 12]),
    np.arange(-24, -8, 0.5)]
blob_i2 = [np.array([3.8]),
    10**np.arange(7., 14.5, 0.5)]
blob_i3 = [np.arange(5, 15.1, 0.1)]
blob_i4 = None
blob_f1 = ['pops[0].LuminosityFunction']
blob_f2 = ['pops[0].fstar']
blob_f3 = None
blob_f4 = None
##
#

base_pars = \
{
 'pop_Tmin{0}': 1e4,
 'pop_Tmin{1}': 'pop_Tmin{0}',
 'pop_model{0}': 'sfe',
 'pop_model{1}': 'sfe',
 'pop_MAR{0}': 'hmf',

 #'pop_sed{0}': 'leitherer1999',
 'pop_kappa_UV{0}': 1.15e-28,

 'pop_fesc{0}': 0.2,

 'pop_ion_src_igm{1}': False,
 'pop_yield{1}': 2.6e39,
 
 'pop_lf_Mmax{1}': 1e15,
 
 'pop_fstar{0}': 'php',
 
 'php_Mfun{0}': 'dpl',
 'php_Mfun_par0{0}': 0.6,
 'php_Mfun_par1{0}': 1e12,
 'php_Mfun_par2{0}': 0.5,
 'php_Mfun_par3{0}': 0.5,
 
 # No dustcorr for now
 'dustcorr_Afun': None,
  
 'problem_type': 101.2,
 
 'photon_counting': True,
 'cgm_initial_temperature': 2e4,
 'cgm_recombination': 'B',
 'clumping_factor': 3.,
 'load_ics': False,
 'smooth_derivative': 10,
 'final_redshift': 5.,
 
 'blob_names': [blob_n1, blob_n2],#, blob_n3, blob_n4],
 'blob_ivars': [blob_i1, blob_i2],#, blob_i3, blob_i4],
 'blob_funcs': [blob_f1, blob_f2],#, blob_f3, blob_f4],
}

# Initialize a fitter object and give it the data to be fit
fitter = ares.inference.FitLuminosityFunction(**base_pars)

pars = \
    [
     'php_Mfun_par0{0}', 'php_Mfun_par1{0}', 
     'php_Mfun_par2{0}', 'php_Mfun_par3{0}',
    ]

is_log = [False, True, False, False]

fitter.parameters = pars
fitter.is_log = is_log

# PRIORS
ps = ares.inference.PriorSet()
ps.add_prior(ares.inference.Priors.UniformPrior(0, 1), 'php_Mfun_par0{0}')
ps.add_prior(ares.inference.Priors.UniformPrior(7, 13),'php_Mfun_par1{0}')
ps.add_prior(ares.inference.Priors.UniformPrior(0, 1), 'php_Mfun_par2{0}')
ps.add_prior(ares.inference.Priors.UniformPrior(0, 1), 'php_Mfun_par3{0}')
fitter.prior_set = ps

# Setup # of walkers and initial guesses for them
fitter.nwalkers = 48

fitter.data = 'bouwens2015'
fitter.redshifts = [3.8]

fitter.jitter = [0.1] * len(pars)

fitter.save_data('test_lf', clobber=True)

fitter.runsim = False
fitter.save_hmf = False

# Run the thing
t1 = time.time()
fitter.run('test_lf', burn=0, steps=20, save_freq=1, clobber=True)
t2 = time.time()

print "Run complete in %.4g minutes.\n" % ((t2 - t1) / 60.)



