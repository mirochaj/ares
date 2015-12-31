"""

test_fitting_21cm.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Nov 16 11:16:08 PST 2015

Description: 

"""


import ares
import numpy as np
import matplotlib.pyplot as pl

# Initialize a fitter object and give it the data to be fit
fitter = ares.inference.FitGlobal21cm(gaussian_model=True)

fitter.turning_points = 'C'

fitter.data = {'gaussian_model': True}
fitter.error = {'C': (0.2, 5.)}

# Give the dimensions of the parameter space names (optional)
fitter.parameters = ['gaussian_A', 'gaussian_nu', 'gaussian_sigma']
fitter.is_log = False

# Setup # of walkers and initial guesses for them
fitter.nwalkers = 100

fitter.jitter = [10, 10, 5]
fitter.guesses = [-100, 70., 10.]
fitter.priors = \
{
 'gaussian_A': ['uniform', -500, 0],
 'gaussian_nu': ['uniform', 40, 120],
 'gaussian_sigma': ['uniform', 0, 20],
}

# Run the thing
fitter.run('test_generic_21cm', steps=100, save_freq=10, clobber=True)

# Read-in the results and make a few plots
anl = ares.analysis.ModelSet('test_generic_21cm')

# Best-fit model
pars = anl.max_likelihood_parameters()

#pl.plot(x, best_fit, color='b', label='best fit')
#pl.legend(fontsize=14)

# Confidence contours
anl.TrianglePlot(anl.parameters, fig=2)
