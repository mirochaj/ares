"""

test_fitting_fcoll.py

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
 'auto_generate_blobs': True,
}

# Initialize fitter
fit = ares.inference.ModelFit(**base_pars)

# Input model: all defaults
sim = ares.simulations.Global21cm()
sim.run()

fit.mu = ares.analysis.Global21cm(sim).turning_points

# Set axes of parameter space
fit.set_axes(['fX', 'Tmin'], is_log=[True]*2)
fit.priors = {'fX': ['uniform', -2., 2.], 'Tmin': ['uniform', 3., 5.]}

# Set errors
fit.set_error(error1d=[0.5, 0.5, 0.5, 5., 5., 5.])

# Defines order of errors
fit.measurement_map = \
    [('B', 0), ('C', 0), ('D', 0),
     ('B', 1), ('C', 1), ('D', 1)]

fit.nwalkers = 8

# Run it!
t1 = time.time()
fit.run(prefix='test_fcoll', steps=50, clobber=True, save_freq=1)
t2 = time.time()

print "Run complete in %.4g minutes.\n" % ((t2 - t1) / 60.)

#