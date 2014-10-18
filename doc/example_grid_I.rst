Simple Parameter Study
======================
Often we want to study how the 21-cm signal changes over a range of parameters. 
We can do so using the :class:`ModelGrid<glorb.search.ModelGrid>` class, 
and use numpy arrays to represent the range of values we're interested in.

Note: you will need to download `ndspace <https://bitbucket.org/mirochaj/ndspace>`_ 
for this to work. It handles the creation and manipulation of N-D model grids.

To begin, import glorb and initialize an instance of the ``ModelGrid`` class:

:: 

    import glorb
    mg = glorb.search.ModelGrid()
    
Let's survey a small 2-D swath of parameter space, varying the X-ray 
normalization parameter and star formation efficiency:

::

    mg.setup(fX=np.linspace(0.1, 0.5, 5), fstar=np.array([0.05, 0.1]))
    
Load-balancing can be very advantageous -- there are a few built-in methods for doing this, 
but more on that later. For this example we'll turn off load balancing since 
we're running in an interactive Python session in serial:
    
::

    mg.load_balance(method=0)
    
Finally, to run the thing:

::

    mg.run(verbose=False)

The main results are stored in an ``ndspace.ModelGrid`` instance, which contains
information about the grid axes:

::

    mg.grid.axes
    ax0, ax1 = mg.grid.axes
    
    ax0.name
    ax0.values
    
    # etc.

And the results:

::
    
    mg.grid.shape

In this case, the grid shape is (5, 2, 2). 5 is the number of ``fX`` values surveyed, 
the middle dimension corresponds to the ``fstar axis``, and the last dimension 
represents the :math:`(z, \delta T_b)` pair for each turning point, 
which can be accessed by name:

::

    mg['B']
    
There are some convenience functions for picking out the results for individual models. 
For example, say you were particularly interested in the case of ``fX=0.2``, 
``fstar=0.1``:

::

    loc = mg.grid.locate_entry({'fX':0.2, 'fstar':0.1})
    mg.grid['B'][loc]
    mg.grid['C'][loc]

To see where the turning points happen, you could make a simple scatter-plot:

::
    
    import matplotlib.pyplot as pl
    
    for pt in ['B', 'C', 'D']:
        pl.scatter(mg.grid[pt][...,0], mg.grid[pt][...,1])
    
If you're interested in variations in ``Tmin``, in which case load-balancing
could be highly advantageous, see :doc:`example_grid_II`.

To save a model grid calculation, do: ::

    mg.grid.to_hdf5('glorb_modelgrid.hdf5')
    
To access it later (and analyze it as we did above), do: ::

    mg = glorb.analysis.ModelGrid('glorb_modelgrid.hdf5')
    
    

    