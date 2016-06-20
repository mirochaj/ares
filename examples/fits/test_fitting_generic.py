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
from ares.inference.ModelFit import ModelFit, LogLikelihood

# Create some fake data
x = np.arange(0, 10, 0.1)
y = np.exp(-(x - 5.)**2 / 2. / 2.**2)
y += np.random.normal(loc=0., scale=0.1, size=len(y))

# Plot it
pl.plot(x, y, label='"data"', drawstyle='steps-mid', color='k')

# Initialize a fitter object and give it the data to be fit
fitter = ModelFit()
fitter.xdata = x
fitter.ydata = y
fitter.error = 0.5 * np.ones_like(y)

# Define the model (a Gaussian)
model_func = lambda x, *pars: pars[0] * np.exp(-(x - pars[1])**2 / 2. / pars[2]**2)

class loglikelihood(LogLikelihood):
    # Inheriting LogLikelihood means mostly that some attribute-setting
    # is taken care of for us.
    def __call__(self, pars):
        
        point = {}
        for i in range(len(self.parameters)):
            point[self.parameters[i]] = pars[i]
        
        lp = self.priors_P.log_prior(point)
        if not np.isfinite(lp):
            return -np.inf, self.blank_blob
        
        model = model_func(self.xdata, *pars)
        
        return -np.sum((self.ydata - model)**2 / 2. / self.error**2), {}

# Give the dimensions of the parameter space and names (optional)
fitter.parameters = ['A', 'mu', 'sigma']
fitter.is_log = False

# PRIORS
ps = ares.inference.PriorSet()
ps.add_prior(ares.inference.Priors.UniformPrior(0.1, 10), 'A')
ps.add_prior(ares.inference.Priors.UniformPrior(0.1, 10),'mu')
ps.add_prior(ares.inference.Priors.UniformPrior(0.1, 10), 'sigma')
fitter.prior_set = ps

# Setup # of walkers and initial guesses for them
fitter.nwalkers = 100

# Set the loglikelihood attribute of the fitter
fitter.loglikelihood = loglikelihood(x, y, fitter.error, fitter.parameters,
    False, None, ps)

# Run the thing
fitter.run('test_generic_mcmc', steps=500, save_freq=50, clobber=True)

# Read-in the results and make a few plots
anl = ares.analysis.ModelSet('test_generic_mcmc')

# Best-fit model
pars = anl.max_likelihood_parameters()

best_fit = map(lambda x: model_func(x, pars['A'], pars['mu'], pars['sigma']), x)
pl.plot(x, best_fit, color='b', label='best fit')
pl.legend(fontsize=14)

# Confidence contours
anl.TrianglePlot(anl.parameters, fig=2)
