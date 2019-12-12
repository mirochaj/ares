:orphan:

Embedding *ARES* in your own code
=================================
If you want to summon *ARES* to generate models within a larger calculation (say, a model grid, MCMC fit, etc.) but for whatever reason do NOT want to use the built-in machinery for such things (no hard feelings!), the procedure should be fairly painless. You can basically follow the same approach as has been outlined in other areas of the documentation:

    * :doc:`example_gs_standard`
    * :doc:`example_litdata` (see bottom for common models)
    
    
Typically, the only hurdle is to correctly supply parameters to new *ARES* simulation objects. *ARES* expects all parameters to be supplied via keyword arguments, so if you have an array of parameter combinations you'd like to run (as in the case of MCMC), you'll have to convert each sub-array to a dictionary first.

For example, say you want to vary the minimum virial temperature of star-forming halos and the normalization of the :math:`L_X`-SFR relation. First, setup a dictionary of parameters that will *not* change from model to model:

::

    import ares
    
    base_kwargs = ares.util.ParameterBundle('mirocha2017:dpl')

.. note :: See :doc:`example_litdata` regarding use of ``mirocha2017:dpl`` models.

Now, let's setup a grid of models to evaluate.

::
    
    # The order here matters: we'll assume below that the first column
    # of values in the model grid correspond to Tmin, the second to rad_yield.
    pars = ['pop_Tmin{0}', 'pop_rad_yield{1}']
    
    # Run models over all combinations of these values
    vals = {'pop_Tmin{0}': [1e3, 1e4, 1e5], 'pop_rad_yield{1}': [1e38, 1e39, 1e40, 1e41]}
    
    grid = []
    for i, val1 in enumerate(vals['pop_Tmin{0}']):
        for j, val2 in enumerate(vals['pop_rad_yield{1}']):
            grid.append([val1, val2])
    
    grid = np.array(grid)        
    
This ``grid`` array has the same shape as an MCMC chain. In practice, you may have such an array that you constructed yourself by some other means. Regardless, once you've got it, you can loop through its elements and run *ARES* simulations via, e.g.,

::

    for i, model in enumerate(grid):
        kw = {par:grid[i][j] for j, par in enumerate(pars)}
        
        kwargs = base_kwargs.copy()
        kwargs.update(kw)
        
        print "Running model #{0}...".format(i)
        sim = ares.simulations.Global21cm(**kwargs)
        sim.run()
        
        # To save the data, could simply use index i to get unique filenames
        sim.save('model_{}'.format(i))
        
This is kind of a silly example because the first step of structuring the parameter grid is completely unnecessary: we could have simply run the simulations within that double for loop. However, the idea is that you might in general have a grid of parameters you setup in some other way, or that you obtained from the outputs of an MCMC.

Note also that in this example you'd be left to parse all the outputs from individual calculations yourself. Not such a terrible thing, but if you're going to run large sets of models, it might be worth using the built-in routines for running big model grids, which automatically collect and distill the information into a format that can be easily analyzed via (you guessed it) other built-in analysis routines. If you still want to do your own thing, that's OK: you may want to eliminate the call to ``sim.save`` above, and extract only the pieces of information you are interested in (from ``sim.history``) and write-out in a format of your choosing.

See :doc:`example_grid` and :doc:`example_grid_analysis` for more information on *ARES*' internal model grid routines.


