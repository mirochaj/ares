"""

test_fitting_tanh.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon May 12 14:19:33 MDT 2014

Description: Can run this in parallel.

"""

import time
import numpy as np
import matplotlib.pyplot as pl
from ares.util.TanhModel import *
from ares.inference import ModelFit

blobs = (['igm_Tk', 'igm_heat', 'cgm_Gamma', 'cgm_h_2', 'Ts', 'Ja', 'dTb'], 
        ['B', 'C', 'D'])

# These go to every calculation
base_pars = \
{
 'final_redshift': 5.,
 'tanh_model': True,
 'inline_analysis': blobs,
}

# Initialize fitter
fit = ModelFit(**base_pars)

# Assume default parameters
fit.set_input_realization(tanh_model=True)

# Set axes of parameter space
is_log = [False]*2

fit.set_axes(['tanh_xz0', 'tanh_xdz'], is_log=is_log)

fit.priors = \
{
 'tanh_xz0': ['uniform', 5., 20.],
 'tanh_xdz': ['uniform', 0.1, 10]
}

fit.nwalkers = 8

# Run it!
t1 = time.time()
fit.run(prefix='test_tanh_1', steps=200, clobber=True, save_freq=1)
t2 = time.time()

print "Run complete in %.4g minutes." % ((t2 - t1) / 60.)

