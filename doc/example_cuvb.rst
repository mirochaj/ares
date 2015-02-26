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
To evolve this kind of radiation background in time, we'll use the 
:class:`ares.simulations.MetaGalacticBackground` class:

:: 

    pars = \
    {
     "source_type": 'star', 
     "source_temperature": 3e4,  # ballpark O-type star
     "spectrum_type": 'bb', 
     "spectrum_Emin": 10.2,
     "spectrum_Emax": 13.6,
     "approx_lwb": False,        # this tells ares to solve the RTE
     "norm_by": 'lw',            
     "Nlw": 1e4,                 # number of LW photons / baryon in star formation
    }

    sim = ares.simulations.MetaGalacticBackground(**pars)
    
This may take a few seconds to initialize.

Then, to calculate the background flux: ::    

    sim.run()

In general, these kinds of calculations could have multiple populations emitting
in different bands. To extract the flux history for a single population (in 
this case the 0th population, by default):

::

    z, E, flux = sim.get_history()
    
This returns three things: first, the redshift points sampled in the history,
second, the photon energies at which the flux was computed, and lastly, the
fluxes themselves. The flux array has shape `(z, E)`, while the energies are
sorted (for numerical reasons) by Lyman-n band. To stitch the energies
into a single array, simply concatenate:

::

    Eflat = np.concatenate(E)      # Energies split by Ly-n bands, stitch together first

Finally, plot up the background spectrum (at the final redshift)

    pl.semilogy(Eflat, flux[-1])
    
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
    