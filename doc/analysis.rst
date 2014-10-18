Built-In Analysis Tools
=======================
ares has a built-in analysis module which contains routines for performing standard
operations like plotting the ionization history, thermal history, and global 21-cm
signature for a given input model. For example, as in the previous example:

::

    import glorb
    sim = glorb.run.Simulation()
    sim.run()
    
    # ...once simulation is complete
    anl = glorb.analysis.Synthetic21cm(sim)
    ax = anl.GlobalSignature()
    
There are some convenience routines to add additional axes that convert redshift
to observed 21-cm frequency, or time since Big Bang, i.e., ::

    anl.add_frequency_axis(ax)
    
and ::
    
    anl.add_time_axis(ax)

These routines can be accessed using the ``freq_ax`` and ``time_ax`` keywords arguments
to ``GlobalSignature`` as well.    
    
If instead you'd like to examine the ionization/thermal evolution: ::

    ax = anl.IonizationHistory()
    ax = anl.TemperatureHistory()
    
or look at the cumulative CMB optical depth: ::

    ax = anl.OpticalDepthHistory()
    
Each of these functions accept standard matplotlib keyword arguments like 
``color``, ``ls``, ``label``, and so on.
    
    

