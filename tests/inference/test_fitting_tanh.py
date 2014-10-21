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
 'tanh_dz': 0.2,
 'inline_analysis': blobs,
}

# Initialize fitter
fit = ModelFit(**base_pars)

# Assume default parameters
sim = ares.simulations.Global21cm(tanh_model=True)
fit.mu = ares.analysis.Global21cm(sim).turning_points

# Set axes of parameter space
fit.set_axes(['tanh_xz0', 'tanh_xdz', 'tanh_Tz0', 'tanh_Tdz'], 
    is_log=[False]*4)

fit.priors = \
{
 'tanh_xz0': ['uniform', 5., 20.],
 'tanh_xdz': ['uniform', 0.1, 10],
 'tanh_Tz0': ['uniform', 5., 20.],
 'tanh_Tdz': ['uniform', 0.1, 10],
}

# Set errors
fit.set_error(error1d=[0.5, 0.5, 0.5, 5., 5., 5.])

# Defines order of errors
fit.measurement_map = \
    [('B', 0), ('C', 0), ('D', 0),
     ('B', 1), ('C', 1), ('D', 1)]

fit.nwalkers = 32

# Run it!
t1 = time.time()
fit.run(prefix='test_tanh', burn=100, steps=1e3, clobber=True, save_freq=10)
t2 = time.time()

print "Run complete in %.4g minutes.\n" % ((t2 - t1) / 60.)

