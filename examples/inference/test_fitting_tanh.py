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
 'tanh_model': True,
 'auto_generate_blobs': True,
}

# Initialize fitter
fit = ares.inference.ModelFit(**base_pars)

# Assume default parameters
fit.mu = np.array([45, 70, 105, -5, -100, 20])

# Set axes of parameter space
fit.set_axes(['tanh_xz0', 'tanh_xdz', 'tanh_Tz0', 'tanh_Tdz'],
    is_log=[False]*4)

# Set priors on model parameters (uninformative)
fit.priors = \
{
 'tanh_xz0': ['uniform', 5., 20.],
 'tanh_xdz': ['uniform', 0.1, 20],
 'tanh_Tz0': ['uniform', 5., 20.],
 'tanh_Tdz': ['uniform', 0.1, 20],
}

# Set errors
fit.set_error(error1d=[0.5, 0.5, 0.5, 5., 5., 5.])

fit.measurement_units = ('MHz', 'mK')

# Defines order of errors
fit.measurement_map = \
    [('B', 0), ('C', 0), ('D', 0),
     ('B', 1), ('C', 1), ('D', 1)]

fit.nwalkers = 128

# Run it!
t1 = time.time()
fit.run(prefix='test_tanh', burn=10, steps=1e3, clobber=True, save_freq=10)
t2 = time.time()

print "Run complete in %.4g minutes.\n" % ((t2 - t1) / 60.)

