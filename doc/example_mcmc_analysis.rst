:orphan:

Analyzing MCMC Calculations
===========================
If you don't yet have a dataset to work with, you can make one by following :doc:`example_mcmc_gs` or :doc:`example_mcmc_lf`. Also, the analysis machinery for MCMC calculations is identical to that used to analyze model grids, so it may also be useful to look at :doc:`example_grid_analysis`.

.. Specialized Analysis: Global 21-cm Signal
.. -----------------------------------------
.. 
.. ::
.. 
..     anl = ares.analysis.ModelSetLF('lffit')
..     
..     mp2 = anl.TrianglePlot(anl.parameters, color_by_like=True, 
..         colors=['g', 'b'], fig=2)
.. 

    

Specialized Analysis: Galaxy Luminosity Functions
-------------------------------------------------
There is yet another specialized class for analyzing fits of LF data: ``ModelSetLF``, which just inherits the standard ``ModelSet`` class and has a few convenience routines for plotting.

First, create an instance of said class:

::

    anl = ares.analysis.ModelSetLF('lffit')
    
Let's have a look at a triangle plot showing constraints on all parameters:    
    
::    
    
    mp = anl.TrianglePlot(anl.parameters, color_by_like=True, 
        colors=['g', 'b'], fig=1)

Now, let's have a look at how our best fit does compared to some data. First, show a few observational datasets (by default, will include whatever exists in ``$ARES/input/litdata`` -- see :doc:`example_litdata` -- at the right redshifts :math:`\pm \texttt{round_z}`):

::

    redshifts = [6,7,8]
    obslf = ares.analysis.GalaxyPopulation()

    mp2 = obslf.MultiPlot(redshifts, round_z=0.4, ncols=3, fig=2)
    
Next, loop over the redshifts of interest and plot the best fit:    

::

    for i, z in enumerate(redshifts):

        # Figure out the index of the axis object for this redshift
        j = obslf.redshifts_in_mp[i]

        # Plot a semi-opaque band representing confidence level `like`
        anl.LuminosityFunction(z, shade_by_like=True,
            ax=mp2.grid[j], color='darkgray', like=0.95)   
        # Plot the best-fit (taken here to be the median)
        anl.LuminosityFunction(z, ax=mp2.grid[j], color='k', ls='-', 
            best_fit='median')

    mp.fix_ticks() # eliminates redundant ticks, scales all axes to span common range
    
If you'd like to visualize individual samples of the posterior, you can do so via, e.g.,

::

    for i, z in enumerate(redshifts):
        j = obslf.redshifts_in_mp[i]
        
        # Notice `samples` keyword argument!
        anl.LuminosityFunction(z, shade_by_like=False, samples=100,
            ax=mp2.grid[j], color='b', alpha=0.1)    
    

To access the best fit parameters directly, 

::

    kw = anl.max_likelihood_parameteris(method='median')

or 

::

    kw = anl.max_likelihood_parameteris(method='maxL')
    
which you can then use to re-run a model if you'd like, e.g.,

::

    pars = anl.base_kwargs.copy()
    pars.update(kw)
    sim = ares.simulations.Global21cm(**pars)

or, if you setup the fit in such a way that only parameters for a single population were supplied (i.e., not all the parameters needed for a ``Global21cm`` calculation were given), you'd instead do 

::

    pop = ares.populations.GalaxyPopulation(**pars)
    
and go on from there.    


