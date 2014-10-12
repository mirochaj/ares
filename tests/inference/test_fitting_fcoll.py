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
}

# Initialize fitter
fit = ModelFit(**base_pars)

# Assume default parameters
fit.set_input_realization()

# Set axes of parameter space
fit.set_axes(['fX'], is_log=[True]*1)

fit.priors = {'fX': ['uniform', -2., 2.]}

fit.nwalkers = 4

# Run it!
t1 = time.time()
fit.run(prefix='test_fcoll', steps=10, clobber=True)
t2 = time.time()

print "Run complete in %.4g minutes." % ((t2 - t1) / 60.)

