"""

test_fitting_fcoll.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon May 12 14:19:33 MDT 2014

Description: Can run this in parallel.

"""

import time
import numpy as np
import matplotlib.pyplot as pl
from ares.inference import ModelFit

blobs = (['igm_Tk', 'igm_heat', 'cgm_Gamma', 'cgm_h_2', 'Ts', 'Ja', 'dTb'], 
        ['B', 'C', 'D'])

# These go to every calculation
base_pars = \
{
 'final_redshift': 8.,
 'inline_analysis': blobs,
 'Tmin{0}': 1e4,
 'source_type{0}': 'star',
 'fstar{0}': 1e-1,
 'Nion{0}': 4e3,
 'Nlw{0}': 9600.,
 'Tmin{1}': 300.,
 'source_type{1}': 'star',
 'is_lya_src{1}': True,
 'is_ion_src_igm{1}': False,
 'is_ion_src_cgm{1}': False, 
 'is_heat_src_igm{1}': False, 
 'fstar{1}': 1e-4,
 'Nion{1}': 3e4,
 'Nlw{1}': 4800.,
}

# Initialize fitter
fit = ModelFit(**base_pars)

# Assume default parameters
fit.set_input_realization(**base_pars)

# Set axes of parameter space
fit.set_axes(['fstar{0}', 'fstar{1}'], is_log=[True]*2)
fit.priors = {'fstar{0}': ['uniform', -3., 0.],
              'fstar{1}': ['uniform', -6., 0.]}

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

