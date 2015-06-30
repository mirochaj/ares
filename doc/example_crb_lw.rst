The Metagalactic Lyman-Werner Background
========================================
One of the main motivations for *ares* was to be able to easily generate
models for the metagalactic background. In this example, we'll focus on the
background near the Lyman-Werner band, which is noteworthy given the
``sawtooth'' modulation (e.g., `Haiman et al. (1997)
<http://adsabs.harvard.edu/abs/1997ApJ...476..458H>`_) caused by intergalactic hydrogen atoms.

In order to model this background, we need to decide on a few main ingredients:

* The spectrum of sources, which can be one of several pre-defined options (like a power-law, ``pl`` or blackbody, ``bb``), or a Python function supplied by the user.
* How the background will evolve with redshift, which could be based on the rate of collapse onto dark matter haloes as a function of time, a parameterized form for the star-formation rate history, or a again, an arbitrary Python function supplied by the user.
* What (if any) approximations we'll make in order to speed-up the calculation, aside from the assumption of a spatially uniform radiation background, which we make implicitly in *ares* throughout.

First things first:

::

    import ares
    import numpy as np
    import matplotlib.pyplot as pl

Now, let's set some parameters that define the properties of the source population:

::
    
    pars = \
    {
     'pop_type': 'galaxy',
     'pop_sfrd': lambda z: 0.1,
     'pop_sed': 'pl',
     'pop_alpha': 1.0,          
     'pop_Emin': 10.18,
     'pop_Emax': 15.,
     'pop_EminNorm': 13.6,
     'pop_EmaxNorm': 1e2,
     'pop_yield': 1e57,
     'pop_yield_units': 'photons/msun',

     # Solution method
     'pop_solve_rte': True,
     'pop_tau_Nz': 400,
     'include_H_Lya': False,

     'initial_redshift': 40,
     'final_redshift': 10,
    }
    
To summarize these inputs, we've got:

* A constant SFRD of :math:`0.1 \ M_{\odot} \ \mathrm{yr}^{-1} \ \mathrm{cMpc}^{-3}`, given by the ``pop_sfrd`` parameter.
* A flat spectrum (power-law with index :math:`\alpha=0`), given by ``pop_sed`` and ``pop_alpha``.
* A yield of :math:`10^{57} \ \mathrm{photons} \ M_{\odot}^{-1}` of star-formation in the :math:`13.6 \leq h\nu / \mathrm{eV} \leq  100` band, set by ``pop_EminNorm``, ``pop_EmaxNorm``, ``pop_yield``, and ``pop_yield_units``.

See :doc:`params_populations` for a complete listing of parameters relevant to :class:`ares.populations.GalaxyPopulation` objects.

Next, let's initialize an :class:`ares.simulations.MetaGalacticBackground` object (which will automatically create an :class:`ares.populations.GalaxyPopulation` instance):

::

    mgb = ares.simulations.MetaGalacticBackground(**pars)

So long as ``verbose=True`` (which it is by default), you should see the following output to the screen:

::

    ##########################################################################
    ####                  Initializer: Galaxy Population                  ####
    ##########################################################################
    #### ---------------------------------------------------------------- ####
    #### Redshift Evolution                                               ####
    #### ---------------------------------------------------------------- ####
    #### SFRD        : parameterized                                      ####
    #### ---------------------------------------------------------------- ####
    #### Radiative Output                                                 ####
    #### ---------------------------------------------------------------- ####
    #### yield (erg / s / SFR) : 3.43977e+39                              ####
    #### EminNorm (eV)         : 13.6                                     ####
    #### EmaxNorm (eV)         : 100                                      ####
    #### ---------------------------------------------------------------- ####
    #### Spectrum                                                         ####
    #### ---------------------------------------------------------------- ####
    #### SED               : pl                                           ####
    #### Emin (eV)         : 1                                            ####
    #### Emax (eV)         : 41.8                                         ####
    #### alpha             : 1                                            ####
    #### logN              : -inf                                         ####
    ##########################################################################
    
The only real difference you may notice is that the units of the yield have been converted to :math:`\mathrm{erg} \ \mathrm{s}^{-1} \ (M_{\odot} \ \mathrm{yr}^{-1})^{-1}`.    
    
To run the thing:

::

    mgb.run()

The results of the calculation, as in any ``ares.simulations`` class, are stored in an attribute called ``history``. Here, we'll use a convenience routine to extract the redshifts, photon energies, and corresponding fluxes (a 2-D array):

::

    z, E, flux = mgb.get_history(flatten=True)

and plot the flux at the final redshift (:math:`z=10`):

::

    pl.semilogy(E, flux[-1], color='k', ls=':')
    
You should see...    
    
By default, *ares* will not do any sort of detailed radiative transfer that accounts for neutral absorption, which is why the background spectrum  To turn that on,

::

    pars2 = pars.copy()
    pars2['pop_sawtooth'] = True
    
    mgb2 = ares.simulations.MetaGalacticBackground(**pars2)
    mgb2.run()
    
    z2, E2, flux2 = mgb2.get_history(flatten=True)
    pl.semilogy(E2, flux2[-1], color='k', ls='--')
    
Compare to the analytic solution, given by Equation A1 in `Mirocha (2014) <http://adsabs.harvard.edu/abs/2014arXiv1406.4120M>`_ (the *cosmologically-limited* solution to the radiative transfer equation)

.. math::
    
    J_{\nu}(z) = \frac{c}{4\pi} \frac{\epsilon_{\nu}(z)}{H(z)} \frac{(1 + z)^{9/2-(\alpha + \beta)}}{\alpha+\beta-3/2} \times \left[(1 + z_i)^{\alpha+\beta-3/2} - (1 + z)^{\alpha+\beta-3/2}\right]

with :math:`\alpha = \beta = 0` (i.e., constant SFRD, flat spectrum), :math:`z=10`, and :math:`z_i=40`,

::

    # Grab the GalaxyPopulation instance
    pop = mgb.pops[0] 
    
    # Compute cosmologically-limited solution
    e_nu = np.array(map(lambda E: pop.Emissivity(10, E), E))
    e_nu *= c / 4. / np.pi / pop.cosm.HubbleParameter(10.) 
    e_nu *= (1. + 10.)**4.5 / -1.5
    e_nu *= ((1. + 40.)**-1.5 - (1. + 10.)**-1.5)
    e_nu *= ev_per_hz
    
    # Plot it
    pl.semilogy(E, e_nu, color='k', ls='-')
    
Add some axis labels:

::

    pl.xlabel(ares.util.labels['E'])
    pl.ylabel(ares.util.labels['flux_E'])
    
When it's all said and done, you should have something like the plot below.

.. figure::  http://casa.colorado.edu/~mirochaj/docs/glorb/basic_star.png
   :align:   center
   :width:   600

   The Lyman-Werner (and below) background at :math:`z=10` that arises from a population of flat spectrum sources.

    