"""

test_fitting_tanh.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon May 12 14:19:33 MDT 2014

Description: Can run this in parallel.

"""

import time, ares
import numpy as np
import matplotlib.pyplot as pl

# These go to every calculation
base_pars = \
{
 'problem_type': 101,
 'tanh_model': True,
 'blob_names': [['tau_e', 'z_B', 'z_C', 'z_D', 'igm_dTb_C', 'igm_dTb_D'], 
    ['cgm_h_2', 'igm_Tk', 'igm_dTb']],
 'blob_ivars': [None, np.arange(6, 21)],
 'blob_funcs': None,
}

# Initialize fitter
fitter = ares.inference.FitGlobal21cm(**base_pars)

fitter.turning_points = list('BCD')

# Assume default parameters
fitter.data = base_pars

# Set axes of parameter space
fitter.parameters = ['tanh_xz0', 'tanh_xdz', 'tanh_Tz0', 'tanh_Tdz']
fitter.is_log = [False]*4

# Set priors on model parameters (uninformative)
ps = ares.inference.PriorSet()
ps.add_prior(ares.inference.Priors.UniformPrior(5., 20.), 'tanh_xz0')
ps.add_prior(ares.inference.Priors.UniformPrior(0.1, 20.), 'tanh_xdz')
ps.add_prior(ares.inference.Priors.UniformPrior(5., 20.), 'tanh_Tz0')
ps.add_prior(ares.inference.Priors.UniformPrior(0.1, 20.), 'tanh_Tdz')
ps.add_prior(ares.inference.Priors.GaussianPrior(0.066, 0.012), 'tau_e')
fitter.prior_set = ps

# Set errors
fitter.error = {tp:[1.0, 5.] for tp in list('BCD')}

fitter.nwalkers = 128

# Run it!
t1 = time.time()
fitter.run(prefix='test_tanh_extrema', burn=10, steps=50, clobber=True, 
    save_freq=10)
t2 = time.time()

print("Run complete in {:.4g} minutes.\n".format((t2 - t1) / 60.))

