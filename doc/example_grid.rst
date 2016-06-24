:origin:

Simple Parameter Study: 2-D Model Grid
======================================
Often we want to study how the 21-cm signal changes over a range of parameters. We can do so using the ModelGrid class, and use numpy arrays to represent the range of values we’re interested in.

Before we start, the few usual imports:

::

    import ares
    import numpy as np
    
Quick Example: :math:`tanh` model for the global 21-cm signal
-------------------------------------------------------------
Before we run a set of models, we need to decide what quantities we’d like to save. For a detailed description of how to do this in general cases, check out :doc:`example_inline_analysis`.

For now, let’s save the redshift and brightness temperature of the global 21-cm emission maximum, which we dub "Turning Point D", and the CMB optical depth,

::

    blobs_scalar = ['z_D', 'dTb_D', 'tau_e']
    
in addition to the ionization, thermal, and global 21-cm histories at redshifts between 5 and 20 (at :math:`\Delta z = 1` increments),

::

    blobs_1d = ['cgm_h_2', 'igm_Tk', 'dTb']
    blobs_1d_z = np.arange(5, 21)
    
.. note :: For a complete listing of ideas for 1-D blobs see :doc:`fields`.
    
Now, we’ll make a dictionary full of parameters that will get passed to every global 21-cm signal calculation. In addition to the blobs, we’ll set ``tanh_model=True`` to speed things up (see next section regarding physical models), and ``problem_type=101``:    

::

    base_pars = \
    {
     'problem_type': 101,
     'tanh_model': True,
     'blob_names': [blobs_scalar, blobs_1d],
     'blob_ivars': [None, blobs_1d_z],
     'blob_funcs': None,
    }
    
and create the ``ModelGrid`` instance,    
    
::

    mg = ares.inference.ModelGrid(**base_pars)
    
At this point we have yet to specify which parameters will define the axes of the model grid. Since we set ``tanh_model=True``, we have 9 parameters to choose from. Let’s take the reionization redshift, ``tanh_xz0``, and duration, ``tanh_xdz``, and sample them over a reasonable redshift interval with a spacing of :math:`\Delta z = 0.1`

::

    z0 = np.arange(6, 12, 0.1)
    dz = np.arange(0.1, 8.1, 0.1)
    
Now, we just set the ``axes`` attribute to a dictionary containing the array of values for each parameter:

::

    mg.axes = {'tanh_xz0': z0, 'tanh_xdz': dz}
    
To run,

::

    mg.run('test_2d_grid', clobber=True, save_freq=100)

To speed things up, you could increase the grid spacing. Or, execute the above in parallel as a Python script (assuming you have MPI and mpi4py installed).

    .. note:: If the model grid doesn’t finish running, that’s OK! Simply    
        re-execute the above command with ``restart=True`` as an 
        additional keyword argument and it will pick up where it left off.
    
To analyze the results, create an analysis instance,    

::

    anl = ares.analysis.ModelSet('test_2d_grid')
    
and, for example, plot the 2-d parameter space with points color-coded by ``tau_e``,

::

    anl.Scatter(anl.parameters, c='tau_e')
    
or instead, the position of the emission maximum with the same color coding:

::

    anl.Scatter(['z_D', 'igm_dTb_D'], c='tau_e')
    
See :doc:`example_grid_analysis` for more information.

More Expensive Models
---------------------
Setting ``tanh_model=True`` sped things up considerably in the previous example. In general, you can run grids varying any *ares* parameters you like, just know that physical models take a few seconds each, whereas the :math:`tanh` model takes much less than a second for one model.

In one particular case -- when ``Tmin`` is one axis of the model grid -- load-balancing can be very advantageous. Just execute the following command before running the grid:

::
    
    mg.LoadBalance(method=1)
    



    
