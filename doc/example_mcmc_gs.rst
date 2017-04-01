:orphan:

Fitting the Global 21-cm Signal
===============================
It's relatively straightforward to call the Markov-Chain Monte Carlo code
`emcee <http://dan.iel.fm/emcee/current/>`_ (`Foreman-Mackey et al. (2013) <http://adsabs.harvard.edu/abs/2013PASP..125..306F>`_),
and perform a fits to:

    - The turning points in the global 21-cm signal. 
    - The galaxy luminosity function
    - Something really cool I haven't even thought of yet!

Here, we'll focus on the global signal application.
    
Fitting Global 21-cm Extrema
----------------------------
The fastest model to fit is one treating the Lyman-:math:`\alpha`, ionization, and thermal histories as a tanh function, so that's what we'll use in this example. 

::

    import numpy as np

    # These go to every calculation
    base_pars = \
    {
     'problem_type': 101,
     'tanh_model': True,
     'blob_names': [['tau_e'], ['cgm_h_2', 'igm_Tk', 'igm_dTb']],
     'blob_ivars': [None, [('z', np.arange(6, 21))]],
     'blob_funcs': None,
    }
    
.. note :: These ``blob_*`` parameters were covered in :doc:`example_grid`, so if you have yet to go through that example, now might be a good time!
    
Now, initialize a fitter:

::   

    import ares
    
    # Initialize fitter
    fitter = ares.inference.FitGlobal21cm(**base_pars)
 
and the turning points to be fit:

::

    fitter.turning_points = True
    
    # Assume default parameters
    fitter.data = {'tanh_model': True}
    
    # Set errors
    fitter.error = {tp:[1.0, 5.] for tp in list('BCD')}
    
    
Now, we set the parameters to be varied in the fit and whether or not to explore their values in log10,

::

    # Set axes of parameter space
    fitter.parameters = ['tanh_xz0', 'tanh_xdz', 'tanh_Tz0', 'tanh_Tdz']
    fitter.is_log = [False]*4
    
as well as the priors on the parameters, which in this case we'll take to be uninformative over a relatively broad range:

::

    from ares.inference import PriorSet
    from ares.inference.Priors import UniformPrior
    
    ps = PriorSet()
    ps.add_prior(UniformPrior(5, 20), 'tanh_xz0')
    ps.add_prior(UniformPrior(0.1, 20), 'tanh_xdz')
    ps.add_prior(UniformPrior(5, 20), 'tanh_Tz0')
    ps.add_prior(UniformPrior(0.1, 20), 'tanh_Tdz')
    
    fitter.prior_set = ps
    
Finally, we set the number of Goodman-Weare walkers 

::

    fitter.nwalkers = 128
    
and run the fit:
      
::    
    
    fitter.run(prefix='test_tanh', burn=10, steps=50, save_freq=10)

This will result in a series of files named ``test_tanh*.pkl``. See the example on :doc:`example_mcmc_analysis` to proceed with inspecting the above dataset.

Fitting Global 21-cm Signal
---------------------------
To fit the entire spectrum, rather than just the turning points, the above example requires only minor modification. 

Whereas previously we set

::

    fitter.turning_points = True

    # Assume default parameters
    fitter.data = {'tanh_model': True}

    # Set errors
    fitter.error = {tp:[1.0, 5.] for tp in list('BCD')}
    
now, we must provide errors at a specified set of frequencies:

::

    fitter.turning_points = False
    fitter.frequencies = np.arange(50, 200) # assumed to be in MHz

    # Assume default parameters
    fitter.data = {'tanh_model': True}

    # Set errors to be a constant 10 mK across the band
    fitter.error = 10. * np.ones_like(fitter.frequencies)
    
That's it!    


