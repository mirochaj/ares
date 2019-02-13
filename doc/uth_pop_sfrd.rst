:orphan:

Models for Star Formation in Galaxies
=====================================
The are a number of different ways to model star formation in *ares*. The method employed is determined by the value of the parameter ``pop_sfr_model``, which can take on any of the following values:

    + ``'fcoll'``
        Relate the global star formation rate density (SFRD) to the rate at which matter collapses into halos above some threshold.
    + ``'sfrd-func'``
        Model the SFRD with a user-supplied function of redshift. 
    + ``'sfe-func'``
        Model the star formation efficiency (SFE) as a function of halo mass and (optionally) redshift.
    + ``'link:ID'``
        Link the SFRD to that of the population with given ID number.
        
Each of these is discussed in more detail below.

``fcoll`` models
~~~~~~~~~~~~~~~~
In this case the SFRD is modeled as:

.. math :: \mathrm{SFRD} = f_{\ast} \bar{\rho}_b^0 \frac{d f_{\mathrm{coll}}}{dt}

where :math:`f_{\ast}` is the efficiency of star formation, :math:`\bar{\rho}_b^0` is the mean baryon density today, and :math:`f_{\mathrm{coll}}` is the fraction of mass in collapsed halos above some threshold.

A basic set of ``'fcoll'`` parameters can be summoned via:

::
    
    import ares
    
    pars = ares.util.ParameterBundle('pop:fcoll')
    
To initialize a population, just do:

::

    pop = ares.populations.GalaxyPopulation(**pars)
    
    # Print SFRD at redshift 20.
    print pop.SFRD(20.)

This will be a very small number because *ares* uses *cgs* units internally, which means the SFRD is in units of :math:`\mathrm{g} \ \mathrm{s}^{-1} \ \mathrm{cm}^{-3}`, with the volume assumed to be co-moving. To convert to the more familiar units of :math:`M_{\odot} \ \mathrm{year}^{-1} \ \mathrm{cMpc}^{-3}`, 

::

    from ares.physics.Constants import rhodot_cgs
    
    print pop.SFRD(20.) * rhodot_cgs
    
.. note :: You can also provide ``pop_Tmax`` (or ``pop_Mmax``) to relate the        
    SFRD to the rate of collapse onto halos above ``pop_Tmin`` *and* below 
    ``pop_Tmax`` (or ``pop_Mmax``). 



``sfrd-func`` models
~~~~~~~~~~~~~~~~~~~~
If ``pop_sfr_model=='sfrd_func'`` you'll need to provide your SFRD function via the ``pop_sfrd`` parameter. You can use a ``ParameterBundle`` if you'd like, though in this case it is particularly short:

::

    pars = ares.util.ParameterBundle('pop:sfrd-func')

A really simple example would be just to make this population have a constant star formation history:

::

    pars['pop_sfrd'] = lambda z: 1e-2
    
However, you could also use a ``ParameterizedHaloProperty`` here (see :doc:`param_populations` for more details). This might be advantageous if, for example, you want to vary the parameters of the SFRD in a model grid or Monte Carlo simulation. 

Let's make a power-law SFRD. For example, the following:

::
    
    pars['pop_sfr_model'] = 'sfrd-func'
    pars['pop_sfrd'] = 'pq'
    pars['pq_func'] = 'pl'
    pars['pq_func_var'] = '1+z'
    pars['pq_func_par0'] = 1e-2
    pars['pq_func_par1'] = 7.
    pars['pq_func_par2'] = -6

sets the SFRD to be

.. math :: \mathrm{SFRD} = 10^{-2} \left(\frac{1 + z}{7} \right)^{-6} M_{\odot} \ \mathrm{year}^{-1} \ \mathrm{cMpc}^{-3}


``sfrd-tab`` models
~~~~~~~~~~~~~~~~~~~
Alternatively, you can supply a lookup table for the SFRD. To do this, modify your parameters as follows:

::

    pars['pop_sfr_model'] = 'sfrd-tab'
    pars['pop_sfrd'] = (z, sfrd)

where ``z`` and ``sfrd`` are arrays you've generated yourself. *ares* will construct an interpolant from these arrays using ``scipy.interpolate.interp1d``, using the method supplied in ``pop_sfrd_interp``. By default, this will be a ``'cubic'`` spline, but you can also supply, e.g., ``pop_sfrd_interp='linear'``.

By default, *ares* assumes your SFRD is in units of :math:`\mathrm{g} \  \mathrm{s}^{-1} \ \mathrm{cm}^{-3}` (co-moving) (corresponding to ``pop_sfrd_units='internal'``), but if you can change this to 'msun/yr/cmpc^3' if you'd prefer the more sensible units of :math:`M_{\odot} \ \mathrm{yr}^{-1} \ \mathrm{cMpc}^{-3}`! In fact, these are the only two options, so as long as ``pop_sfrd_units != 'internal'``, *ares* assumes the :math:`M_{\odot} \ \mathrm{yr}^{-1} \ \mathrm{cMpc}^{-3}` units.


``sfe-func`` models
~~~~~~~~~~~~~~~~~~~
Rather than parameterizing the SFRD directly, it is possible to parameterize the star formation efficiency as a function of halo mass and redshift, and integrate over the halo mass function in order to obtain the global SFRD.

Grab a few parameters to begin:

::

    pars = ares.util.ParameterBundle('sfe-func')
    
This set of parameters assumes a double power-law for the SFE as a function of halo mass with sensible values for the parameters. To create a population instance, as per usual,

::

    pop = ares.populations.GalaxyPopulation(**pars)
    
To test the SFE model, 

::

    import numpy as np
    import matplotlib.pyplot as pl
    
    Mh = np.logspace(7, 13, 100)
    pl.loglog(Mh, pop.SFE(z=10, M=Mh))
    
    
and the SFRD:

::

    pop.SFRD(10.)
    
    
See :doc:`example_pop_galaxy` for more information about this.    

``link`` models
~~~~~~~~~~~~~~~
Say you're running a simulation with multiple populations and, while their radiative properties are different, you want them to have the same star formation histories. To be concrete, let's make a simple ``fcoll`` population and tag it with an identification number:

::
    
    pop0 = ares.util.ParameterBundle('pop:fcoll')
    pop0.num = 0

Now, let's make a second population with the same star-formation model:

::
    
    pop1 = {'pop_sfr_model{1}': 'link:0'}
    
    # Add together
    pars = pop0 + pop1
    
The ``'link:0'`` means "link to population #0". So, if we initialize a simulation with both populations, e.g.,

::

    sim = ares.simulations.Global21cm(**pars)
    
and compare their SFRDs, they should be equal:

::

    sim.pops[0].SFRD(20.) == sim.pops[1].SFRD(20.)
    
.. note :: The ``pop_sfr_model`` for population #0 could be anything in the example above. However, only the SFRD function will be shared between the two populations -- all other attributes of populations #0 and #1 will be completely independent. 
    
    
