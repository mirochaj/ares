Advanced Parameter Study
========================
In this example, we'll follow the same procedure as in the `Simple Parameter Study' example, but add a few dimensions, and take advantage of some advanced features. First, as always, import glorb and initialize a Gridded21cm instance:

:: 

    import glorb
    import numpy as np
    
    mg = glorb.search.ModelGrid()
    
Let's survey a 2-D swath of parameter space, varying the X-ray normalization 
parameter and ``Tmin``, the minimum virial temperature of star-forming halos:

::

    mg.setup(fX=np.linspace(0.1, 0.5, 3), Tmin=np.logspace(3, 4, 3))
    
This is a case where load-balancing is very helpful. The ``Tmin`` dimension of 
this parameter space requires some significant overhead at the outset of each 
calculation - but once this has been done once, we can simply store that 
information and use it again and again. Because of this, load-balancing over 
the ``Tmin`` dimension of a model grid makes lots of sense:

::

    mg.load_balance(method=1)
    
Finally, to run the thing:

::

    mg.run(fn='advanced_param_study.hdf5', thru='D')

The ``fn`` keyword argument is a filename that our results will automatically be
saved to (in HDF5), and the ``thru`` keyword argument indicates the end-point of
each simulation. In this case, its set to turning point D (roughly indicates 
start of EoR), but other options are ``'B'``, ``'C'``, and ``'trans'``. This is useful if 
we're only interested in the pre-reionization era (e.g., ``thru='C'``) or the 
the first stars feature (e.g., ``thru='B'``) for example, in which case we don't 
want to waste time computing the entire reionization history.

Note: you can pass additional keyword arguments to ``mg.run``, which will be
used for each individual model in the grid.

