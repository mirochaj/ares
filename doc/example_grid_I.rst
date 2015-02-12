Simple Parameter Study
======================
Often we want to study how the 21-cm signal changes over a range of parameters. 
We can do so using the :class:`ModelGrid<glorb.search.ModelGrid>` class, 
and use *numpy* arrays to represent the range of values we're interested in.

To begin,

:: 

    import ares
    import numpy as np
    
Before we run a set of models, we need to decide what quantities we'd like
to save for each model. Anything stored in the ``history`` attribute of an
``ares.simulations.Global21cm`` instance is fair game: see :doc:`fields` for
more information. We also must supply a series of redshifts
at which to save the quantities of interest.

Let's save the following quantities:

* 21-cm brightness temperature, ``dTb``.
* Spin temperature, ``Ts``.
* Kinetic temperature, ``igm_Tk``.
* HII region volume filling factor, ``cgm_h_2``.
* Neutral fraction in the bulk IGM, ``igm_h_1``.
* Heating rate in the IGM, ``igm_heat_h_1``.
* Volume-averaged ionization rate, or rate of change in ``cgm_h_2``, ``cgm_Gamma_h_1``.

i.e., ::

    fields = ['z', 'dTb', 'Ts', 'igm_Tk', 'cgm_h_2', 'igm_h_1', 'igm_heat_h_1', 'cgm_Gamma_h_1']

We'll save each of these quantities at the three extrema in the global 21-cm
signal (turning points B, C, and D), and a few other redshifts of interest.
    
::
    
    redshifts = ['B', 'C', 'D', 6, 8, 10, 12]
    
.. note :: You can also pass ``auto_generate_blobs=True`` as a keyword 
    argument, which will (try to) figure out all relevant quantities of 
    interest and save them at a default series of redshifts.
        
Now, initialize a ``ModelGrid`` instance: 

::

    base_kwargs = \
    {
     'inline_analysis': [fields, redshifts], 
    }

    mg = ares.inference.ModelGrid(**base_kwargs)
    
``base_kwargs`` will be passed to every model in the grid. Note the ``inline_analysis``
key: this tells ares to automatically record the values of our fields of interest
at the specified redshifts.    

.. note :: You can also pass ``auto_generate_blobs`` or ``inline_analysis`` 
    as keyword arguments to the :class:`Global21cm<ares.simulations.Global21cm>` 
    class, which will then automatically run the requested analysis upon 
    completion of a simulation and save the results to the ``blobs`` attribute.
    To conveniently recover a particular quantity, see the ``extract_blob`` 
    method in :class:`Global21cm<ares.simulations.Global21cm>`. If you forget
    what quantities and redshifts are available, see the ``blob_names`` and
    ``blob_redshifts`` attributes of the same class. 
    
Now, let's survey a small 2-D swath of parameter space, varying the X-ray 
normalization parameter and star formation efficiency:

::
    
    mg.set_axes(fX=np.linspace(0.1, 0.5, 5), fstar=np.array([0.05, 0.1]))
    
Load-balancing can be very advantageous -- there are a few built-in methods for doing this, 
but more on that later. For this example we'll turn off load balancing since 
we're running in an interactive Python session in serial:
    
::

    mg.LoadBalance(method=0)
    
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

.. note :: The above can be run in parallel as a Python script, so long as you have `MPI <http://www.open-mpi.org/>`_ and `mpi4py <http://mpi4py.scipy.org>`_ installed.

It's easiest to analyze the results using a built-in analysis module, which 
will automatically retrieve the data in all files with the given prefix:
    
::
    
    anl = ares.analysis.ModelSet('test_model_grid')

To see where the absorption trough occurs (turning point C), you could make a simple scatter-plot
showing the redshift and brightness temperature for each model at that point:

::
    
    ax = anl.Scatter(x='nu', y='dTb', z='C')

To see the where the emission signal occurs on the same axes, 

::

    ax = anl.Scatter(x='nu', y='dTb', z='D', color='r')
    
If you're interested in variations in ``Tmin``, in which case load-balancing
could be highly advantageous, see :doc:`example_grid_II`. For more examples
of the built-in analysis routines, check out :doc:`example_grid_analysis`.


    