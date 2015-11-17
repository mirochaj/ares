"""

test_fitting_lf.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Nov 16 14:59:43 PST 2015

Description: 

"""

import ares, time
import numpy as np
import matplotlib.pyplot as pl

base_pars = \
{
 'pop_Tmin{2}': 1e4,
 'pop_model{2}': 'ham',
 'pop_Macc{2}': 'mcbride2009',

 'pop_lf_Mstar{2}[4.9]': -22,
 'pop_lf_pstar{2}[4.9]': 1e-5,
 'pop_lf_alpha{2}[4.9]': -2,
 'pop_lf_z{2}': [4.9],
 
 'pop_lf_mags{2}': 'bouwens2015',
 'pop_kappa_UV{2}': 1.15e-28,
 'pop_EminNorm{2}': 13.6,
 'pop_EmaxNorm{2}': 24.6,
 'pop_Emin{2}': 13.6,
 'pop_Emax{2}': 24.6,
 'pop_yield{2}': 4000.,
 'pop_yield_units{2}': 'photons/baryon',
 'pop_fesc{2}': 0.2,
 'cgm_initial_temperature': 2e4,
 'cgm_recombination': 'B',
 'clumping_factor': 3.,
 'load_ics': False,
 'need_for_speed': True,
}

# Initialize a fitter object and give it the data to be fit
fitter = ares.inference.FitLuminosityFunction(**base_pars)

fitter.parameters = ['pop_lf_Mstar{2}[4.9]', 'pop_lf_pstar{2}[4.9]', 
    'pop_lf_alpha{2}[4.9]']
fitter.is_log = [False, True, False]

fitter.priors = \
{
 'pop_lf_Mstar{2}[4.9]': ['uniform', -24, -18],
 'pop_lf_pstar{2}[4.9]': ['uniform', -6, 0.],
 'pop_lf_alpha{2}[4.9]': ['uniform', -4, -1],
}

# Setup # of walkers and initial guesses for them
fitter.nwalkers = 100

fitter.data = 'bouwens2015'
fitter.redshifts = 4.9

fitter.guesses = 'bouwens2015'

fitter.save_data('test_lf', clobber=True)

fitter.runsim = False

# Run the thing
t1 = time.time()
fitter.run('test_lf', burn=0, steps=10, save_freq=10, clobber=True)
t2 = time.time()

print "Run complete in %.4g minutes.\n" % ((t2 - t1) / 60.)
