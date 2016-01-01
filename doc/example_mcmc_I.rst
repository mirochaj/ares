:orphan:

Running *ares* with *emcee*
===========================
It's relatively straightforward to call the Markov-Chain Monte Carlo code
`emcee <http://dan.iel.fm/emcee/current/>`_ (`Foreman-Mackey et al. (2013) <http://adsabs.harvard.edu/abs/2013PASP..125..306F>`_),
and perform a fits to:

    - The turning points in the global 21-cm signal. 
    - The galaxy luminosity function
    - Something really cool I haven't even thought of yet!

Some examples below.

Generic MCMC Fitting
--------------------





::

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
    
    
    
    
Fitting Global 21-cm Extrema
----------------------------
The fastest model to fit is one treating the Lyman-:math:`\alpha`, ionization, and thermal histories as a tanh function:

::

    # These go to every calculation
    base_pars = \
    {
     'problem_type': 101,
     'tanh_model': True,
     'blob_names': [['tau_e'], ['cgm_h_2', 'igm_Tk', 'igm_dTb']],
     'blob_ivars': [None, np.arange(6, 21)],
     'blob_funcs': None,
    }
    
    
Now, initialize a fitter:

::   
    
    # Initialize fitter
    fit = ares.inference.ModelFit(**base_pars)
 
and the turning points to be fit:

::
    
    # Assume default parameters
    fitter.data = base_pars
    
    # Set errors
    fitter.error = {tp:[1.0, 5.] for tp in list('BCD')}
    
    
Finally, we set the parameters to be varied in the fit and their priors, which
in this case we'll take to be uninformative over a relatively broad range:

::

    # Set axes of parameter space
    fitter.parameters = ['tanh_xz0', 'tanh_xdz', 'tanh_Tz0', 'tanh_Tdz']
    fitter.is_log = [False]*4
    
    # Set priors on model parameters (uninformative)
    fit.priors = \
    {
     'tanh_xz0': ['uniform', 5., 20.],
     'tanh_xdz': ['uniform', 0.1, 20],
     'tanh_Tz0': ['uniform', 5., 20.],
     'tanh_Tdz': ['uniform', 0.1, 20],
    }

    # Set the number of Goodman-Weare walkers 
    fit.nwalkers = 128
    
To finally run it, 
      
::    
    
    fit.run(prefix='test_tanh', burn=10, steps=50, save_freq=10)

This will result in a series of files named ``test_tanh*.pkl``. 

Fitting Global 21-cm Signal
---------------------------


Fitting the Galaxy Luminosity Function
--------------------------------------

