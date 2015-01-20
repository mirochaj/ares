Built-In Analysis Tools
=======================
ares has a built-in analysis module which contains routines for performing standard
operations like plotting the ionization history, thermal history, and global 21-cm
signature for a given input model. For example, 

::

    import ares
    sim = ares.simulations.Global21cm()
    sim.run()
    
    # ...once simulation is complete
    anl = ares.analysis.Global21cm(sim)
    ax = anl.GlobalSignature()
    
If instead you'd like to examine the ionization/thermal evolution: ::

    ax2 = anl.IonizationHistory(fig=2)
    ax3 = anl.TemperatureHistory(fig=3)
    
or look at the cumulative CMB optical depth: ::

    ax4 = anl.OpticalDepthHistory(fig=4)
    
Each of these functions accept standard matplotlib keyword arguments like 
``color``, ``ls``, ``label``, and so on.
    
    

