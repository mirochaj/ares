:orphan:

Fitting Galaxy Luminosity Functions
-----------------------------------
If you've already had a look at :doc:`example_mcmc_gs`, the approach detailed below will look familiar: we'll define a dataset to be fit, a set of parameters that each model requires, and a set of parameters allowed to vary in the fit (and priors that tell us how much they are allowed to vary).

One notable distinction between this example and the last is that the "blobs" we are interested in are two dimensional (e.g., we're generally interested in the galaxy luminosity function as a function of both magnitude and redshift), and so we must pay some attention to setting up the calculation. In this example, we'll keep track of the luminosity function and the star formation efficiency. The latter is the thing our parameters describe, so so while we could re-generate the SFE later (for each element of the chain) at very little extra computational cost, we may as well save it. We could also simply re-generate the LF later, but that is a slightly less trivial cost, so it'll save some time to track it as well.

OK, each of these quantities has a different independent variable, but we may as well track them at a common set of redshifts:

::

    # Independent variables
    redshifts = np.array([3, 3.8, 4, 4.9, 5, 5.9, 6, 6.9, 7, 7.9, 8])
    MUV = np.arange(-28, -8.8, 0.2)
    Mh = np.logspace(7, 13, 61)

    # blob 1: the LF. Give it a name, and the function needed to calculate it.
    blob_n1 = ['galaxy_lf']
    blob_i1 = [('z', redshifts), ('x', MUV)]
    blob_f1 = ['LuminosityFunction']
   
    # blob 2: the SFE. Same deal. 
    blob_n2 = ['fstar']
    blob_i2 = [('z', redshifts), ('Mh', Mh)]
    blob_f2 = ['fstar']
  
.. note :: For the independent variables, we must also supply the name of the argument (positional or keyword) expected by the provided function.
    
Stick this all in a dictionary:

::
    
    blob_pars = \
    { 
     'blob_names': [blob_n1, blob_n2],
     'blob_ivars': [blob_i1, blob_i2],
     'blob_funcs': [blob_f1, blob_f2],
     'blob_kwargs': [None] * 2,
    }
    
Note that the ``blob_f?`` variables contain string representations of functions. This is important! In :doc:`example_mcmc_gs`, we didn't have to do this because we only tracked common blobs that live in the ``history`` attribute of the ``ares.simulations.Global21cm`` class (*ARES* knows to first look for blobs in the ``history`` attribute of simulation objects). So, dealing with 2-D blobs requires some knowledge of what's happening in the code. For example, the above will only work if ``LuminosityFunction`` accepts redshift and UV magnitude **in that order**. Also, we had to know that this method is attached to the object stored in the ``pops[0]`` attribute of a simulation object.

Now, let's make our master dictionary of parameters, with one important addition:
        
::

    import ares

    base_pars = ares.util.ParameterBundle('mirocha2017:base').pars_by_pop(0, True)
    base_pars.update(blob_pars)
    
    # This is important!
    base_pars['pop_calib_lum'] = None
    
The ``pop_calib_lum`` parameter tells *ARES* the :math:`1600\AA` luminosity per unit star formation conversion used to derive the input SFE parameters (or at another wavelength through `pop_calib_wave` parameter). This can be useful, for example, if you'd like to vary the parameters of a stellar population (e.g., the metallicity ``pop_Z``) *without* impacting the luminosity function. Of course, when we're fitting the LF, the whole point to allow parameter variations to affect the LF, which is why we must turn it off by hand here.
    
.. note:: By default, *ARES* does not apply a dust correction. This can be useful, for example, if you want to generate a single physical model and study the effects of dust after the fact (see :doc:`example_pop_galaxy`). However, when fitting data, we must make a choice about the dust correction ahead of time since each evaluation of the likelihood will depend on it.
    
OK, now let's set the free parameters and priors:
    
::

    free_pars = \
      [
       'pq_func_par0[0]',
       'pq_func_par1[0]', 
       'pq_func_par2[0]',
       'pq_func_par3[0]',
      ]
    
    is_log = [True, True, False, False]
    
    from distpy import DistributionSet
    from distpy import UniformDistribution
    
    ps = DistributionSet()
    ps.add_distribution(UniformDistribution(-3, 0.), 'pq_func_par0[0]')
    ps.add_distribution(UniformDistribution(9, 13),  'pq_func_par1[0]')
    ps.add_distribution(UniformDistribution(0, 2),   'pq_func_par2[0]')
    ps.add_distribution(UniformDistribution(-2, 0),  'pq_func_par3[0]')
    
    
Some initial guesses (optional: will draw initial walker positions from priors by default):

::

    guesses = \
    {
     'pq_func_par0[0]': -1,
     'pq_func_par1[0]': 11.5,
     'pq_func_par2[0]': 0.5,
     'pq_func_par3[0]': -0.5,
    }
    
Initialize the fitter object:

::
            
    # Initialize a fitter object and give it the data to be fit
    fitter_lf = ares.inference.FitGalaxyPopulation(**base_pars)
    
    # The data can also be provided more explicitly
    fitter_lf.data = 'bouwens2015'
        
Now, in earlier versions of *ARES*, we would have set a few other attributes (which we'll now do below) and then executed ``fitter.run`` with some keyword arguments. But, now, to enable multi-wavelength fitting, we first create a master fitter object:

::

    fitter = ares.inference.ModelFit(**base_pars)
    fitter.add_fitter(fitter_lf)
    
    # Establish the object to which we'll pass parameters
    from ares.populations.GalaxyCohort import GalaxyCohort
    fitter.simulator = GalaxyCohort
    
and then set remaining attributes that establish the free parameters, initial guesses for walkers, number of walkers, etc.,

::    
    
    # A few speed-ups
    fitter.save_hmf = True  # cache HMF for a speed-up!
    fitter.save_psm = True  # cache source SED model (e.g., BPASS, S99)
    
    # Setting this flag to False will make ARES generate new files for each checkpoint. 
    # 2-D blobs can get large, so this allows us to just download a single
    # snapshot or two if we'd like (useful if running on remote machine)
    fitter.checkpoint_append = False    
    
    fitter.parameters = free_pars
    fitter.is_log = is_log
    fitter.prior_set = ps
    
    # In general, the more the merrier (~hundreds)
    fitter.nwalkers = 16
    
    fitter.jitter = [0.1] * len(fitter.parameters)
    fitter.guesses = guesses
    
    # Run the thing
    fitter.run('test_lfcal', burn=0, steps=10, save_freq=1, clobber=True)

This will only take a few minutes to run, but the results will be very crude. Increase the number of walkers, steps, and perhaps add a burn-in for better results.

.. note :: To simultaneously fit luminosity functions and other quantities, 
    one can create separate ``fitter`` objects and simply add them to the fit 
    using the ``fitter.add_fitter`` method, which is essentially just a list    
    of objects that have their own likelihoods.

To see if things are working in the right direction, let's have a quick look at the crude initial results. First, create an analysis instance:

::

    anl = ares.analysis.ModelSet('test_lfcal')
        
and now, let's look at the reconstructed luminosity function, which will tell us if (i) our blobs are being correctly written out to disk, and (ii) if the parameter space is truly being surveyed (if not, all MCMC samples of the LF will be identical).

Since we don't expect the calculation to have converged yet, let's just look at the raw LF samples rather than confidence intervals:

::

    ax = anl.ReconstructedFunction('galaxy_lf', ivar=[6, None], samples='all', color='b', alpha=0.01)
    
    ax.set_yscale('log')
    
To compare to observational data quickly, do 

::

    gpop = ares.analysis.GalaxyPopulation()
    
    # Plot any data within dz=0.1 of z=6
    gpop.PlotLF(6, ax=ax, round_z=0.1)
    ax.set_ylim(1e-9, 1)
    
Hopefully there's agreement at the :math:`\sim`few order-of-magnitude level!
     
See :doc:`example_mcmc_analysis` for general instructions for dealing with the outputs of MCMC calculations. 
