The Metagalactic X-ray Background
=================================
In previous example we've calculated the UV background spectrum at a single
redshift. Here, we'll compute the X-ray background evolution over time, and fast!

::

    import glorb
    import numpy as np
    import matplotlib.pyplot as pl

    # Parameters for population of stellar mass BHs
    params = \
    {
     'source_type': 'bh',
     'Tmin': 1e4,
     'spectrum_type': 'mcd',
     'source_mass': 10.,
     'approx_xray': False,      # Solve the RTE!
     'load_tau': True,          # Look in $GLORB/input/optical_depth for lookup table
     'redshift_bins': 400,      # Sample redshift interval with this many points
    }

    # Initialize RadiationBackground instance
    rad = glorb.evolve.RadiationBackground(**params)
    
    # Compute X-ray flux at all redshifts and photon energies
    z, E, fluxes = rad.XrayBackground()
    
Now, we have the cosmic X-ray background flux at all redshifts, ``z``, and photon
energies, ``E``. So, the variable ``fluxes`` above is a 2-D array with dimensions
``(len(z), len(E))``. If we wanted to plot the background spectrum at a few
redshifts we could do:

::

    for i in range(0, 400, 100):
        pl.loglog(E, fluxes[i], label=r'$z=%.3g$' % z[i])
    
    pl.legend()
    pl.xlabel(glorb.util.labels['E']) 
    pl.ylabel(glorb.util.labels['flux'])

=============================================
Computing the Heating/Ionization Rate Density
=============================================
With fluxes in hand, we can compute the heating rate density and/or
ionization rate density at each redshift straightforwardly:

::

    heat = np.zeros_like(z)
    ioniz = np.zeros_like(z)    

    for i, redshift in enumerate(z):
        heat[i] = rad.igm.HeatingRate(redshift, xray_flux=fluxes[i])
        ioniz[i] = rad.igm.IonizationRateIGM(redshift, xray_flux=fluxes[i])
    
Then, plot the results via:     ::
                        
    pl.semilogy(z, heat)
    
or ::
    
    pl.semilogy(z, ioniz)
    
These values could be used as input to cosmological simulations, just beware 
that by default the values returned by ``HeatingRate`` and ``IonizationRateIGM``
are rates, not rate coefficients, i.e., they have units of :math:`\mathrm{erg} \ \mathrm{s}^{-1} \ \mathrm{cm}^{-3}`
for the heating rate density, and :math:`\mathrm{s}^{-1}` for the ionization
rate density.
    
============================
Tabulating the Optical Depth    
============================    
The above example relied on a pre-existing table of the IGM optical depth over
redshift and photon energy, hence the parameter ``load_tau``, which tells glorb
to go looking in ``$GLORB/input/optical_depth`` for lookup tables. This technique
was outlined originally in Appendix C of `Haardt & Madau (1996) <http://adsabs.harvard.edu/abs/1996ApJ...461...20H>`_.

The shape of the lookup table is defined by the minimum and maximum redshift
(10 and 40 by default), the number of redshift bins used to sample that
interval, ``redshift_bins``, the minimum and maximum photon energies (0.2 and
30 keV by default), and the number of photon energies (determined iteratively
from the redshift and energy intervals and the value of ``redshift_bins``).

To make optical depth tables of your own, see ``$GLORB/examples/generate_optical_depth_tables.py``.
By default, glorb generates tables assuming the IGM is fully neutral, but that
is not required. See Section 3 of `Mirocha (2014) <http://adsabs.harvard.edu/abs/2014MNRAS.443.1211M>`_
for more discussion of this technique.

A more complete example can be found in ``$GLORB/tests/test_cxrb_generator.py``.

===================
Alternative Methods
===================
The technique outlined above is the fastest way to integrate the cosmological
radiative transfer equation (RTE), but it assumes that we can tabulate the 
optical depth ahead of time. What if instead we wanted to study the radiation background in a
decreasingly opaque IGM? Well, we can solve the RTE at several photon energies
in turn: ::

    E = np.logspace(2.5, 4.5, 100)
    
To determine the background intensity at :math:`z=10` due to the same BH population
as above, we could do something like: ::

    # Function describing evolution of IGM ionized fraction with respect to redshift
    # (fully ionized for all time in this case, meaning IGM is optically thin)
    xofz = lambda z: 1.0

    # Compute flux at z=10 and each observed energy due to emission from 
    # sources at 10 <= z <= 20.
    F = [rad.AngleAveragedFlux(10., nrg, zf=20., xavg=xofz) for nrg in E]

    pl.loglog(E, F)
    
You'll notice that computing the background intensity is much slower when
we do not pre-compute the IGM optical depth.    

Let's compare this to an IGM with evolving ionized fraction: :: 
    
    # Here's a function describing the ionization evolution for a scenario
    # in which reionization is halfway done at z=10 and somewhat extended.
    xofz2 = lambda z: glorb.util.xHII_tanh(z, zr=10., dz=4.)
    
    # Compute fluxes
    F2 = [rad.AngleAveragedFlux(10., nrg, zf=20., xavg=xofz2) for nrg in E]
    
    # Plot results
    pl.loglog(E, F2)
    
    # Add some nice axes labels
    pl.xlabel(glorb.util.labels['E'])
    pl.ylabel(glorb.util.labels['flux'])    
    
Notice how the plot of ``F2`` has been hardened by neutral absorption in the IGM!
    
