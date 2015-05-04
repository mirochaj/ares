Running *ares* with *emcee*
===========================
It's relatively straightforward to call the Markov-Chain Monte Carlo code
`emcee <http://dan.iel.fm/emcee/current/>`_ (`Foreman-Mackey et al. (2013) <http://adsabs.harvard.edu/abs/2013PASP..125..306F>`_),
and perform a fit to the turning points in the global 21-cm signal. 

.. note :: Development of a generalized fitter is currently underway.

The fastest model to fit is that of a tanh, as in Harker et al. (2015).

::

    # These go to every likelihood evaluation
    base_pars = \
    {
     'tanh_model': True,
     'auto_generate_blobs': True,
    }
    
Now, initialize a fitter:

::   
    
    # Initialize fitter
    fit = ares.inference.ModelFit(**base_pars)
 
and the turning points to be fit:

::
    
    # Positions of the turning points (nu_B, nu_C, nu_D, T_B, T_C, T_D)
    fit.mu = np.array([45, 70, 105, -5, -100, 20])
    
    # Set errors
    fit.set_error(error1d=[0.5, 0.5, 0.5, 5., 5., 5.])
    
    # Tells fitter that all frequencies are in MHz, all temperatures in mK.
    fit.measurement_units = ('MHz', 'mK')
    
    # Defines order of errors
    fit.measurement_map = \
        [('B', 0), ('C', 0), ('D', 0),
         ('B', 1), ('C', 1), ('D', 1)]
    
Finally, we set the parameters to be varied in the fit and their priors, which
in this case we'll take to be uninformative over a relatively broad range:

::

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

    # Set the number of Goodman-Weare walkers 
    fit.nwalkers = 128
    
To finally run it, 
      
::    
    
    fit.run(prefix='test_tanh', burn=10, steps=50, save_freq=10)

This will result in a series of files named ``test_tanh*.pkl''. 


