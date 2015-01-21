Advanced Parameter Study
========================
In this example, we'll follow the same procedure as in the :doc:`example_grid_I` 
example, but add a few dimensions and take advantage of some advanced
features. It'll be advantageous to write the following as a script (i.e., not
in an interactive Python session) so that we can execute it in parallel. Call 
it ``ares_2d_grid.py``:

:: 

    import ares
    import numpy as np

As in :doc:`example_grid_I`, we'll save the redshift, 21-cm brightness temperature, and spin 
temperature at the redshifts corresponding to extrema in the global signal (which
we refer to as turning points B, C, and D):

::

    fields = ['z', 'dTb', 'Ts']
    redshifts = ['B', 'C', 'D']

and now, initialize a ``ModelGrid`` instance: 

::

    base_kwargs = \
    {
     'inline_analysis': [fields, redshifts], 
    }

    mg = ares.inference.ModelGrid(**base_kwargs)    
    
Let's again survey a 2-D swath of parameter space, varying the X-ray normalization 
parameter and now ``Tmin``, the minimum virial temperature of star-forming halos:

::

    mg.set_axes(fX=np.linspace(0.1, 0.5, 3), Tmin=np.logspace(3, 4, 3))
    
This is a case where load-balancing is very helpful. The ``Tmin`` dimension of 
this parameter space requires some significant overhead at the outset of each 
calculation - but once this has been done once, we can simply store that 
information and use it again and again. Because of this, load-balancing over 
the ``Tmin`` dimension of a model grid makes lots of sense:

::

    mg.load_balance(method=1)

Finally, to run the thing:

::

    mg.run('advanced_param_study')		

To run this as a script, back in the terminal invoke the script with ``mpirun`` 
(here, with 4 cores) ::

    mpirun -np 4 python ares_2d_grid.py

All the usual analysis routines still apply.

