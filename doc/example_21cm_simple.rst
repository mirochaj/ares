Two-Zone IGM Model for the Global 21-cm Signal
==============================================
To begin, first import ares:

:: 

    import ares
    
::

To generate a model of the global 21-cm signal, we need to use the 
``ares.simulations.Global21cm``  class. With no arguments, default parameter 
values will be used:

::
    
    sim = ares.simulations.Global21cm()
    
See the :doc:`parameters` page for a listing of parameters that can be passed
to ``ares.simulations.Global21cm`` as keyword arguments.

Since a lot can happen before we actually 
start solving for the evolution of IGM properties (e.g., initializing radiation
sources, tabulating the collapsed fraction evolution and constructing splines 
for interpolation, tabulating the optical depth, etc.), initialization and 
execution of calculations are separate. To run the simulation, we do:

::

    sim.run()
    
The main results are stored in the attribute ``sim.history``, which is a dictionary
containing the evolution of many quantities with time. To look at the results,
you can access these quantities directly:

::

    import matplotlib.pyplot as pl
    
    pl.plot(sim.history['z'], sim.history['dTb'])

Or, you can pass the ``Global21cm`` instance to an analysis class, which
contains convenience routines for common plots of interest:
    
::
   
    anl = ares.analysis.Global21cm(sim)
    ax = anl.GlobalSignature()
    
.. figure::  http://casa.colorado.edu/~mirochaj/docs/ares/basic_21cm.png
   :align:   center
   :width:   600
   
   One possible realization for the global 21-cm signal.
        
If you'd like to save the results to disk, do something like: 

::

    sim.save('test_21cm')
    
which saves the contents of ``sim.history`` at all time snapshots. 

.. note :: The default format for output files is ``pkl``, though ASCII (e.g., ``.txt``), ``.npz``, and ``.hdf5`` are also supported. Use the optional keyword argument ``suffix``.

To read results from disk, you can supply a filename to ``ares.analysis.Global21cm`` 
rather than a ``ares.simulations.Global21cm`` instance if you'd like, e.g., :: 

    anl = ares.analysis.Global21cm(history='test_21cm.pkl')    

See :doc:`analysis` for more information about readily available analysis 
routines.

DIY Parameter Study
-------------------
To do simple parameter study, you could do something like:

::

    ax = None
    for fX in [0.2, 1.]:
        for fstar in [0.05, 0.1]:
            sim = ares.simulations.Global21cm(fX=fX, fstar=fstar)
            sim.run()

            # Object for common analysis routines
            anl = ares.analysis.Global21cm(sim)
            
            # Plot the global signal
            ax = anl.GlobalSignature(ax=ax, 
                label=r'$f_X=%.2g, f_{\ast}=%.2g$' % (fX, fstar))
                
                
    ax.legend(loc='lower right', fontsize=14) 
    pl.draw()           
                
.. figure::  http://casa.colorado.edu/~mirochaj/docs/ares/ares_simple_param_study.png
   :align:   center
   :width:   600

   Four realizations of the global 21-cm signal, varying the normalization of
   the :math:`L_X`-SFR relation and the star formation efficiency.
                
See :doc:`example_grid_I` for examples of how to run and analyze large grids
of models more easily.


            
            

    