The Metagalactic UV Background
==============================
If you haven't yet looked at the :doc:`example_crb_lw` example, that might be a good place to start as the setup here is very similar, and as a result, we'll skip over a few of the details. 

.. note :: By ''UV background'' here we really mean the *ionizing* background,  
    which, granted, is a little confusing given that the Lyman-Werner band is 
    technically in the UV spectrum as well. This distinction between LW and UV      
    is adopted throughout the code. Sorry about that!
    
The main difference between the ionizing background and the LW background is that the latter is unaffected by the bound-free opacity of the IGM, only experiencing the ''sawtooth modulation'' associated with bound-bound absorption in the Lyman series. The ionizing background has a sawtooth modulation of its own, in the band between the HeII Ly-:math:`\alpha` line and the HeII ionization threshold, :math:`40.8 \leq h\nu / \mathrm{eV} \leq 54.4`.

This is the topic of extensive study in the last :math:`\sim 20` years, e.g., in

* `Haardt & Madau (1996) <http://adsabs.harvard.edu/abs/1996ApJ...461...20H>`_
* `Haardt & Madau (2012) <http://adsabs.harvard.edu/abs/2012ApJ...746..125H>`_
* Several others!

First things first:

::

    import ares
    import numpy as np
    import matplotlib.pyplot as pl

Now, let's set some parameters that define the properties of the source population (very similar to those set in the :doc:`example_crb_lw` example):


::

    # Initialize radiation background
    pars = \
    {
     # Source properties
     'pop_type': 'galaxy',
     'pop_sfrd': lambda z: 0.1,
     'pop_sed': 'pl',
     'pop_alpha': 1.0,
     'pop_Emin': 13.6,
     'pop_Emax': 1e2,
     'pop_EminNorm': 13.6,
     'pop_EmaxNorm': 1e2,
     'pop_yield': 1e57,
     'pop_yield_units': 'photons/msun',

     # Solution method
     'pop_solve_rte': True,
     'pop_tau_Nz': 400,
     'include_H_Lya': False,

     'sawtooth_nmax': 8,
     'pop_sawtooth': True,

     'initial_redshift': 7.,
     'final_redshift': 3.,
    }
    
To summarize these inputs, we've got :

* A constant SFRD of :math:`0.1 \ M_{\odot} \ \mathrm{yr}^{-1} \ \mathrm{cMpc}^{-3}`, given by the ``pop_sfrd`` parameter.
* A flat spectrum (power-law with index :math:`\alpha=0`), given by ``pop_sed`` and ``pop_alpha``.
* A yield of :math:`10^{57} \ \mathrm{photons} \ M_{\odot}^{-1}` of star-formation in the :math:`13.6 \leq h\nu / \mathrm{eV} \leq  100` band, set by ``pop_EminNorm``, ``pop_EmaxNorm``, ``pop_yield``, and ``pop_yield_units``.
* The emission now extends from the Lyman-limit all the way up to 100 eV, which is set by ``pop_Emin`` and ``pop_Emax``.

See :doc:`params_populations` for a complete listing of parameters relevant to :class:`ares.populations.GalaxyPopulation` objects.
    
Initialize the simulation object:

::

    mgb = ares.simulations.MetaGalacticBackground(**pars)

Now, let's run the thing:

::

    mgb.run()

We'll pull out the evolution of the background just as we did in the :doc:`example_crb_lw` example:

::

    z, E, flux = mgb.get_history(flatten=True)

and plot up the result:

::

    from ares.physics.Constants import erg_per_ev

    pl.semilogy(E, flux[-1] * E * erg_per_ev, color='k')
    pl.xlabel(ares.util.labels['E'])
    pl.ylabel(ares.util.labels['flux_E'])
    
You should be able to see the LW sawtooth at the left edge of the plot, and a new sawtooth due to the HeII Lyman series at :math:`40.8 \leq h\nu / \mathrm{eV} \leq 54.4`.

The Opacity of the Clumpy IGM
------------------------------
This is not currently implemented. Check back soon!

    
Recombination Emissivity
------------------------
This is not currently implemented. Check back soon!
    