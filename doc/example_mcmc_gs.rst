:orphan:

Fitting the Global 21-cm Signal
===============================
It's relatively straightforward to call the Markov-Chain Monte Carlo code
`emcee <http://dan.iel.fm/emcee/current/>`_ (`Foreman-Mackey et al. (2013) <http://adsabs.harvard.edu/abs/2013PASP..125..306F>`_),
and perform a fits to:

    - The global 21-cm signal. 
    - The galaxy luminosity function.
    - Something really cool I haven't even thought of yet!

Here, we'll focus on the global signal application.
    
Fitting the Global 21-cm Spectrum
---------------------------------
A fast model that yields semi-realistic global 21-cm signals is one which treats the Lyman-:math:`\alpha`, ionization, and thermal histories as tanh functions (see `Harker et al. 2016 <http://adsabs.harvard.edu/abs/2016MNRAS.455.3829H>`_), so that's what we'll use in this example. 

First, define the parameters that remain unchanged from model to model (mostly abstracted away by ``problem_type=101`` and ``tanh_model=True`` settings), including some metadata blobs:

::

    import numpy as np

    # These go to every calculation
    base_pars = \
    {
     'problem_type': 101,
     'tanh_model': True,
     'blob_names': [['tau_e', 'z_C', 'dTb_C'], ['cgm_h_2', 'igm_Tk', 'dTb']],
     'blob_ivars': [None, [('z', np.arange(6, 31))]],
     'blob_funcs': None,
    }
    
.. note :: These ``blob_*`` parameters were covered in :doc:`example_grid`, so if you have yet to go through that example, now might be a good time!
    
Now, initialize a fitter:

::   

    import ares
    
    # Initialize fitter
    fitter_gs = ares.inference.FitGlobal21cm()
        
and the signal to be fit (just a -100 mK Gaussian signal at 80 MHz with :math:`\sigma=20` MHz for simplicity):

::
    
    fitter_gs.frequencies = freq = np.arange(40, 200) # MHz
    fitter_gs.data = -100 * np.exp(-(80. - freq)**2 / 2. / 20.**2)
    
    # Set errors
    fitter_gs.error = 20. # flat 20 mK error
    
At this point, we're ready to initialize the master fitter:

::

    fitter = ares.inference.ModelFit(**base_pars)
    fitter.add_fitter(fitter_gs)
    fitter.simulator = ares.simulations.Global21cm

In general, we can add more fitters in this fashion -- their likelihoods will simply be summed.
    
Now, we set the parameters to be varied in the fit and whether or not to explore their values in log10:

::

    # Set axes of parameter space
    fitter.parameters = ['tanh_J0', 'tanh_Jz0', 'tanh_Jdz', 'tanh_Tz0', 'tanh_Tdz']
    fitter.is_log = [True] + [False] * 4
    
as well as the priors on the parameters, which in this case we'll take to be uninformative over a relatively broad range (to do this we need Keith Tauscher's `distpy <https://bitbucket.org/ktausch/distpy>`_ package):

::

    from distpy import DistributionSet
    from distpy import UniformDistribution
    
    ps = DistributionSet()
    ps.add_distribution(UniformDistribution(-3, 3), 'tanh_J0')
    ps.add_distribution(UniformDistribution(5, 20), 'tanh_Jz0')
    ps.add_distribution(UniformDistribution(0.1, 20), 'tanh_Jdz')
    ps.add_distribution(UniformDistribution(5, 20), 'tanh_Tz0')
    ps.add_distribution(UniformDistribution(0.1, 20), 'tanh_Tdz')
    
    fitter.prior_set = ps
    
Finally, we set the number of Goodman-Weare walkers 

::

    fitter.nwalkers = 16  # In general, the more the merrier (~hundreds)
    
and run the fit:
      
::    
    
    # Do a quick burn-in and then run for 50 steps (per walker)
    fitter.run(prefix='test_tanh', burn=10, steps=50, save_freq=10)

This will result in a series of files named ``test_tanh*.pkl``. See the example on :doc:`example_mcmc_analysis` to proceed with inspecting the above dataset.

.. note :: For a simple model like the tanh, this fitting will be slower to run through *ARES* due to the overhead of initializing objects and performing the analysis (like finding extrema) in real time. For more sophisticated models, this overhead is dwarfed by the cost of each simulation, and for the complex blobs, the built-in machinery for I/O is very useful. If all you're interested in is phenomenological fits, then it'll be much faster to simply write your own wrappers around *emcee*.

Hopefully you recover a signal with a peak at 80 MHz and -100 mK, but beware that this will be nowhere near converged, so the plots won't be pretty unless you increase the number of steps, walkers, or both.

