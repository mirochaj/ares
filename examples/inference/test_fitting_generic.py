"""

test_mcmc.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Nov 16 07:57:53 PST 2015

Description: 

"""

import ares
import numpy as np
import matplotlib.pyplot as pl
from ares.inference.ModelFit import ModelFit

# Create some fake data
x = np.arange(0, 10, 0.1)
y = np.exp(-(x - 5)**2 / 2. / 1.**2)
y += np.random.normal(loc=0., scale=0.1, size=len(y))

# Plot it
pl.plot(x, y, label='"data"')

# Initialize a fitter object and give it the data to be fit
fitter = ModelFit()

fitter.xdata = x
fitter.ydata = y
fitter.error = 0.5 * np.ones_like(y)

# Define the model (a Gaussian)
model = lambda x, *pars: pars[0] * np.exp(-(x - pars[1])**2 / 2. / pars[2]**2)

class loglikelihood:
    def __init__(self, xdata, ydata, error, model):
        self.xdata = xdata
        self.ydata = ydata
        self.error = error
        self.model = model
        
    def __call__(self, pars):
        
        model = self.model(self.xdata, *pars)
        
        return -np.sum((self.ydata - model)**2 / 2. / self.error**2), {}

# Give the dimensions of the parameter space names (optional)
fitter.parameters = ['A', 'mu', 'sigma']
fitter.is_log = False

# Setup # of walkers and initial guesses for them
fitter.nwalkers = 100

fitter.jitter = 0.25
fitter.guesses = [1., 5., 1.]

# Set the loglikelihood attribute
fitter.loglikelihood = loglikelihood(x, y, fitter.error, model)

# Run the thing
fitter.run('test_generic_mcmc', steps=500, save_freq=50, clobber=True)

# Read-in the results and make a few plots
anl = ares.analysis.ModelSet('test_generic_mcmc')

# Best-fit model
pars = anl.max_likelihood_parameters()

best_fit = map(lambda x: model(x, pars['A'], pars['mu'], pars['sigma']), x)
pl.plot(x, best_fit, color='b', label='best fit')
pl.legend(fontsize=14)

# Confidence contours
anl.TrianglePlot(anl.parameters, fig=2)
