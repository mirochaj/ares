The Metagalactic UV Background
============================================
In the previous examples, we saw examples of how to initialize stellar and BH
populations, e.g., a stellar population:

:: 

    import ares
    
    pop = ares.populations.StellarPopulation(Tmin=1e4, fstar=0.1)
    
Such populations will give rise to global radiation backgrounds. In this example,
we'll focus on the UV background that arises from a stellar population.
    
============================
The "sawtooth" UV Background
============================
The background spectrum between the Lyman-:math:`\alpha` line and the Lyman-limit
exhibits a series of absorption features due to Lyman series absorption, first
illustrated by `Haiman et al. (1997) <http://adsabs.harvard.edu/abs/1997ApJ...476..458H>`_.

::
    
    # Initialize a radiation background
    rad = ares.solvers.UniformBackground(pop=pop)
    
Note that we need not initialize a :class:`StellarPopulation<ares.populations.StellarPopulation>` 
object first -- we can instead pass keyword arguments directly to the 
:class:`StellarPopulation<ares.evolve.UniformBackground>` class, e.g.:

:: 

    params = \
    {
     "source_type": 'star', 
     "source_temperature": 3e4,
     "spectrum_type": 'bb', 
     "spectrum_Emin": 1., 
     "spectrum_Emax": 13.6,
     "approx_lwb": False,        # this tells ares we'll need to solve the RTE
     "discrete_lwb": False,
     "norm_by": 'lw', 
     "Nlw": 1e4,
    }
    
    rad = ares.solvers.UniformBackground(**params)
    
Then, to calculate the background flux: ::    

    import numpy as np

    # Setup function to compute background intensity at redshift 30  
    flux = lambda EE: rad.AngleAveragedFlux(z=30, E=EE, energy_units=True)

    # Focus on Lyman-Werner-ish band
    E = np.linspace(8., 13.6, 500)  # energies in eV

    # Compute background and plot it
    pl.semilogy(E, map(flux, E))
    
    # Make some nice axes labels
    pl.xlabel(ares.util.labels['E'])
    pl.ylabel(ares.util.labels['flux_E'])
        
.. figure::  http://casa.colorado.edu/~mirochaj/docs/glorb/basic_star.png
   :align:   center
   :width:   600

   The Lyman-Werner (and below) background at :math:`z=30` that arises from a population
   of O stars. The dashed line shows the solution obtained if Lyman series absorption
   is neglected. See ares/tests/solvers/test_lwrb_generator.py for a more complete example.
        
The keyword argument ``energy_units`` converts the fluxes to units of 
:math:`\text{erg} \ \text{s}^{-1} \ \text{cm}^{-2} \ \text{Hz}^{-1} \ \text{sr}^{-1}`.
By default, fluxes are returned in units of :math:`\text{s}^{-1} \ \text{cm}^{-2} \ \text{Hz}^{-1}\ \text{sr}^{-1}`.
    
The dashed line in the above figure shows the solution obtained neglecting Lyman
series absorption. It can be obtained by enforcing an optically thin medium with
the optional keyword argument ``tau``:

::

    flux = lambda EE: rad.AngleAveragedFlux(z=30, E=EE, energy_units=True, tau=0.0)
    
    pl.semilogy(E, map(flux, E), ls='--')    
    