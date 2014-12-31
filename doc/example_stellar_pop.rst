Stellar Populations
===================
In these examples, we will initialize a stellar population object, defined
by the minimum virial temperature of halos in which stars form, :math:`T_{\text{min}}`, 
the star formation efficiency, :math:`f_{\ast}`, and perhaps other things.

To begin, import glorb and initialize an instance of the :class:`StellarPopulation<glorb.populations.StellarPopulation>` class:

:: 

    import glorb
    pop = glorb.populations.StellarPopulation(Tmin=1e4, fstar=0.1)
    
Once initialized, there are several class methods available to compute the star-formation rate density (SFRD) and emissivity (in the UV and X-ray):
    
::

    z = 20.
    pop.SFRD(z)                          # [g / ccm**3 / s]
    pop.XrayLuminosityDensity(z)         # [erg / ccm**3 / s]
    pop.LymanWernerLuminosityDensity(z)  # [erg / ccm**3 / s]

The star formation rate density is given by:

.. math::

  \dot{\rho}_{\ast} = \bar{\rho}_{b,0} f_{\ast} \frac{d f_{\text{coll}}}{dt}
 
where :math:`\bar{\rho}_{b,0}` is the mean baryon density today, :math:`f_{\ast}` is
the star formation efficiency, and :math:`f_{\text{coll}}` is the fraction of gas
in collapsed haloes. :math:`f_{\text{coll}}` can be computed by integrating over
the halo mass function at masses above the corresponding minimum virial temperature.
    
Note: class methods always return values in cgs units, and when applicable, 
volume densities are assumed to be in co-moving units (in the comments above, 
"ccm" stands for "co-moving centimeters").

To convert to more recognizable units, use conversion factors from rt1d:

::

    from rt1d.physics.Constants import rhodot_cgs, cm_per_mpc
    pop.SFRD(z) * rhodot_cgs                              # [Msun / cMpc**3 / yr]
    pop.XrayLuminosityDensity(z) * cm_per_mpc**3          # [erg / cMpc**3 / s]
    
where Msun is solar masses, and cMpc is used to denote co-moving Megaparsecs.


============
Stellar SEDs
============
By default, stellar and black hole populations are defined by an ionizing
luminosity density, but we can also treat their 
spectral energy distribution in detail. For example, we could create a population of 
stars whose SED is a blackbody:

:: 

    import glorb

    # Parameters defining (roughly) an O/B type star
    params = \
     {
      "source_type": 'star', 
      "source_temperature": 3e4, 
      "spectrum_type": 'bb', 
      "spectrum_Emin": 1., 
      "spectrum_Emax": 1e2,
      "approx_lwb": False, 
      "norm_by": 'lw', 
      "Nlw": 1e4,
     }

    # Create Population instance
    pop = glorb.populations.StellarPopulation(**params)
                                 
The ``approx_lwb`` keyword argument tells StellarPopulation that we'll be treating
the UV spectrum of this population in detail. To verify this, access the rs attribute
(which is short for ``radiation source'' to indicate that it is an rt1d.sources.RadiationSource instance):

::

    import numpy as np
    import matplotlib.pyplot as pl
    
    E = np.linspace(1., 13.6, 500)  # energies in eV
    F = map(pop.rs.Spectrum, E)
    
    pl.plot(E, F)  # should look like a blackbody!
    
``pop.rs.Spectrum`` is a function that returns the specific luminosity at input
energy E, and is normalized such that the integral from Emin to Emax is 1.

To investigate the UV background that arises from such a population, 
see :doc:`example_cuvb`.

