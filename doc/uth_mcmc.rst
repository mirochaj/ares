:orphan:

A generic MCMC example
======================
In the example below, we'll show how to write your own likelihood function in the style required by *ARES*.

First, some fake data to work with:

::

    import numpy as np

    x = np.arange(0, 10, 0.1)
    y = np.exp(-(x - 5)**2 / 2. / 1.**2)
    
    # Add some "noise"
    y += np.random.normal(loc=0., scale=0.1, size=len(y))

Now, initialize a fitter object and give it the data to be fit:
    
::

    fitter = ModelFit()
    
    fitter.xdata = x
    fitter.ydata = y
    fitter.error = 0.5 * np.ones_like(y)
    
and define the model (a Gaussian)

::

    model = lambda x, *pars: pars[0] * np.exp(-(x - pars[1])**2 / 2. / pars[2]**2)

The most important part is defining the log-likelihood function, which we'll stick in the ``__call__`` method of a class:

::
    
    class loglikelihood(object):
        def __init__(self, xdata, ydata, error, model):
            self.xdata = xdata
            self.ydata = ydata
            self.error = error
            self.model = model
            
        def __call__(self, pars):
            model = self.model(self.xdata, *pars)
            return -np.sum((self.ydata - model)**2 / 2. / self.error**2), {}

As long as your loglikelihood has the attributes ``xdata``, ``ydata``, and ``error``, the ``ModelFit`` class will be able to use it.

::
    
    # Give the dimensions of the parameter space names (optional)
    fitter.parameters = ['A', 'mu', 'sigma']
    fitter.is_log = False
    
    # Setup # of walkers and initial guesses for them
    fitter.nwalkers = 100
        
    # Set the loglikelihood attribute
    fitter.loglikelihood = loglikelihood(x, y, fitter.error, model)
    
To set priors, use the ``PriorSet`` class:

::    
    
    ps = ares.inference.PriorSet()
    ps.add_prior(UniformPrior(0, 5), 'A')
    ps.add_prior(UniformPrior(2, 10), 'mu')
    ps.add_prior(UniformPrior(0.1, 5), 'sigma')
    fitter.prior_set = ps
    
And finally, to run the thing:

::

    fitter.run('test_generic_mcmc', steps=500, save_freq=50, clobber=True)
    
    