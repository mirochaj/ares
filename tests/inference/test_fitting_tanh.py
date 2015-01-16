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

blobs = (['igm_Tk', 'igm_heat_h_1', 'cgm_Gamma_h_1', 'cgm_h_2', 
          'Ts', 'Ja', 'z', 'dTb', 'tau_e'], 
        ['B', 'C', 'D', 40])

# These go to every calculation
base_pars = \
{
 'final_redshift': 5.,
 'tanh_model': True,
 'tanh_dz': 0.2,
 'track_extrema': True,
 'inline_analysis': blobs,
}

ref_pars = \
{
'tanh_Tz0': 8.,
'tanh_Tdz': 4.,
'tanh_xz0': 10.,
'tanh_xdz': 2.,
}

# Initialize fitter
fit = ares.inference.ModelFit(**base_pars)

# Assume default parameters
#sim = ares.simulations.Global21cm(tanh_model=True, **ref_pars)
#fit.mu = ares.analysis.Global21cm(sim).turning_points
fit.mu = np.array([48.807, 71.462, 109.105, -3.838, -105.978, 15.742])

# Set axes of parameter space
fit.set_axes(['tanh_xz0', 'tanh_xdz', 'tanh_Tz0', 'tanh_Tdz'],
    is_log=[False]*4)

fit.priors = \
{
 'tanh_xz0': ['uniform', 5., 20.],
 'tanh_xdz': ['uniform', 0.1, 20],
 'tanh_Tz0': ['uniform', 5., 20.],
 'tanh_Tdz': ['uniform', 0.1, 20],
 'tau_e': ['gaussian', 0.08, 0.01, 40]
}

# Set errors
fit.set_error(error1d=[0.5, 0.5, 0.5, 5., 5., 5.])

fit.measurement_units = ('MHz', 'mK')

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

