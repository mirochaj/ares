:orphan:

Non-Equilibrium Chemistry
=========================
This example shows some of the inner-workings of the chemical network and solver using a simple hydrogen-only test problem in an isothermal medium.

To begin, first import a few things:

:: 

    import ares
    import numpy as np
    import matplotlib.pyplot as pl
    
    
Let's initialize a grid of 64 cells:

::
        
    # Initialize grid object
    grid = ares.static.Grid(grid_cells=64)
    
So far, this ``grid`` object only knows how many cells it has. To give it
some physical properties, we'll call several setter routines:

::    

    # Set initial conditions
    grid.set_physics(isothermal=True)
    grid.set_chemistry(include_He=False)
    grid.set_density(nH=1.)
    grid.set_ionization()  
    grid.set_temperature(np.logspace(3, 5, 64))
    
The above commands initialize the grid to be isothermal, composed of hydrogen
only, with a density of 1 hydrogen atom per cubic centimeter, initialized to 
be neutral and with temperatures between :math:`10^3 \leq T /\ \mathrm{K} \leq 10^5`.

To see how the ion fractions evolve with time, we can pass this grid off to
the chemistry solver:

::  

    # Initialize chemistry network / solver
    chem = ares.solvers.Chemistry(grid, rt=False)

    # Compute rate coefficients (only need to do this once; isothermal)
    chem.chemnet.SourceIndependentCoefficients(grid.data['Tk'])

Now, to actually run the thing:

::

    # To compute timestep
    timestep = ares.util.RestrictTimestep(grid)

    # Set initial time-step and maximum allowed change
    data = grid.data
    dt = ares.physics.Constants.s_per_myr / 1e3
    dt_max = 1e2 * ares.physics.Constants.s_per_myr
    t = 0.0
    tf = ares.physics.Constants.s_per_gyr

    # Initialize progress bar [optional]
    pb = ares.util.ProgressBar(tf)
    pb.start()

    # Start calculation
    while t < tf:
        pb.update(t)
        
        # Evolve system for time dt
        data = chem.Evolve(data, t=t, dt=dt)
        t += dt 

        # Limit time-step based on maximum rate of change in grid quantities
        new_dt = timestep.Limit(chem.chemnet.q, chem.chemnet.dqdt)

        # Limit to factor of 2x increase in timestep
        dt = min(new_dt, 2 * dt)

        # Impose maximum timestep
        dt = min(dt, dt_max)

        # Make sure we end at t == tf
        dt = min(dt, tf - t)

    pb.finish()   
    
All of this work is done for you each time you call ``ares.simulations.Global21cm`` and ``ares.simulations.RaySegment``.    
    
To visualize the results:

::     

    ax = pl.subplot(111)        
    ax.loglog(T, data['h_1'], color='k', ls='-')
    ax.loglog(T, data['h_2'], color='k', ls='--')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$T \ (\mathrm{K})$')
    ax.set_ylabel('Species Fraction')
    ax.set_ylim(1e-4, 2)
    pl.draw()    






    