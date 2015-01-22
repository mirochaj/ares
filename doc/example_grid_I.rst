Simple Parameter Study
======================
Often we want to study how the 21-cm signal changes over a range of parameters. 
We can do so using the :class:`ModelGrid<glorb.search.ModelGrid>` class, 
and use *numpy* arrays to represent the range of values we're interested in.

To begin,

:: 

    import ares
    
Before we run a set of models, we need to decide what quantities we'd like
to save for each model. Anything stored the ``history`` attribute of an
``ares.simulations.Global21cm`` instance is fair game: see :doc:`fields` for
more information. We also must supply a series of redshifts
at which to save the quantities of interest.

Let's just save the redshift, 21-cm brightness temperature, and spin 
temperature at the redshifts corresponding to extrema in the global signal (which
we refer to as turning points B, C, and D):
    
::

    fields = ['z', 'dTb', 'Ts']
    redshifts = ['B', 'C', 'D']
    
.. note :: The list of redshifts can include numerical values as well.    
    
and now, initialize a ``ModelGrid`` instance: 

::

    base_kwargs = \
    {
     'inline_analysis': [fields, redshifts], 
    }

    mg = ares.inference.ModelGrid(**base_kwargs)
    
``base_kwargs`` will be passed to every model in the grid. Note the ``inline_analysis``
key: this tells ares to automatically record the values of our fields of interest
at the specified redshifts.    
    
Now, let's survey a small 2-D swath of parameter space, varying the X-ray 
normalization parameter and star formation efficiency:

::

    mg.set_axes(fX=np.linspace(0.1, 0.5, 5), fstar=np.array([0.05, 0.1]))
    
Load-balancing can be very advantageous -- there are a few built-in methods for doing this, 
but more on that later. For this example we'll turn off load balancing since 
we're running in an interactive Python session in serial:
    
::

    mg.load_balance(method=0)
    
To verify the properties of the model grid, we can access the names and values
of its axes:

::

    mg.grid.axes
    ax0, ax1 = mg.grid.axes
    
    ax0.name
    ax0.values
    
    # etc.
    
Finally, to run the thing

::

    mg.run(prefix='test_model_grid')

The main results are stored in a series of files with the prefix ``test_model_grid``.

.. note :: If the model grid doesn't finish running, that's OK! Simply re-execute the above command and supply ``restart=True`` as an additional keyword argument, and it will pick up where it left off.

It's easiest to analyze the results using a built-in analysis module, which 
will automatically retrieve the data in all files:
    
::
    
    anl = ares.analysis.ModelSet('test_model_grid')

To see where the absorption trough occurs (turning point C), you could make a simple scatter-plot
showing the redshift and brightness temperature for each model at that point:

::
    
    ax = anl.Scatter(x='z', y='dTb', z='C')

To see the where the emission signal occurs on the same axes, 

::

    ax = anl.Scatter(x='z', y='dTb', z='D', color='r')
    
If you're interested in variations in ``Tmin``, in which case load-balancing
could be highly advantageous, see :doc:`example_grid_II`.


    