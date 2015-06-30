The Metagalactic X-ray Background
=================================
In this example, we'll compute the Meta-Galactic X-ray background over a
series of redshifts at high redshifts (:math:`10 \leq z \leq 40):

::

    zi, zf = (40, 10)
    
    # Initialize radiation background
    pars = \
    {
     'source_type': 'bh',
     'sfrd': lambda z: 0.01 / (1. + z)**3.,
     'spectrum_type': 'pl',
     'spectrum_alpha': -1.5,
     'spectrum_Emin': 2e2,
     'spectrum_Emax': 3e4,
     'spectrum_EminNorm': 2e2,
     'spectrum_EmaxNorm': 5e4,
     'approx_xrb': False,
     'redshifts_xrb': 400,
     'initial_redshift': zi,
     'final_redshift': zf,
    }
    
Now, to initialize a calculation:

::    

    import ares

    sim1 = ares.simulations.MetaGalacticBackground(tau_xrb=True, **pars)
    sim1.run()
    
    z1, E1, flux1 = sim1.get_history()
    
and plot it up at the last snapshot:

::

    import matplotlib.pyplot as pl
    
    pl.loglog(E1, flux1[-1])
    
For comparison, force the intergalactic medium to optically thin at all 
redshifts:

::
    
    sim2 = ares.simulations.MetaGalacticBackground(tau_xrb=False, **pars)
    sim2.run()
    
    z2, E2, flux2 = sim2.get_history()
    pl.loglog(E2, flux2[-1])
        
It's worth emphasizing that we have the cosmic X-ray background flux at all 
redshifts, ``z``, and photon energies, ``E``. So, the ``flux``variables above 
are 2-D arrays with dimensions ``(len(z), len(E))``. If we wanted to plot 
the background spectrum at a few redshifts we could do:

::

    for i in range(0, 400, 100):
        pl.loglog(E1, flux1[i], label=r'$z=%.3g$' % z1[i])
    
    pl.xlabel(ares.util.labels['E']) 
    pl.ylabel(ares.util.labels['flux'])
    
The behavior at low photon energies (:math:`h\nu \lesssim 0.5 \ \mathrm{keV}`)
is due to poor redshift resolution, which leads to overestimates of the 
flux. This is a trade made for speed in solving the cosmological
radiative transfer equation, discussed in detail in Section 3 of 
`Mirocha (2014) <http://adsabs.harvard.edu/abs/2014arXiv1406.4120M>`_. For more
accurate calculations, increase the value of `redshifts_xrb`, just know that 
you'll need to create new lookup tables for the optical depth, or compute
it on-the-fly via the `dynamic_tau` parameter (not yet implemented!), as the 
optical depth lookup tables that ship with *ares* use `redshifts_xrb=400`
as a default.

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
redshift and photon energy, hence the parameter ``discrete_xrb``, which tells ares
to go looking in ``$ARES/input/optical_depth`` for lookup tables. This technique
was outlined originally in Appendix C of `Haardt & Madau (1996) <http://adsabs.harvard.edu/abs/1996ApJ...461...20H>`_.

The shape of the lookup table is defined by the minimum and maximum redshift
(10 and 40 by default), the number of redshift bins used to sample that
interval, ``redshift_bins``, the minimum and maximum photon energies (0.2 and
30 keV by default), and the number of photon energies (determined iteratively
from the redshift and energy intervals and the value of ``redshift_bins``).

To make optical depth tables of your own, see ``$ARES/examples/generate_optical_depth_tables.py``.
By default, ares generates tables assuming the IGM is fully neutral, but that
is not required. See Section 3 of `Mirocha (2014) <http://adsabs.harvard.edu/abs/2014MNRAS.443.1211M>`_
for more discussion of this technique.

A more complete example can be found in ``$ARES/tests/test_cxrb_generator.py``.

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
    xofz2 = lambda z: ares.util.xHII_tanh(z, zr=10., dz=4.)
    
    # Compute fluxes
    F2 = [rad.AngleAveragedFlux(10., nrg, zf=20., xavg=xofz2) for nrg in E]
    
    # Plot results
    pl.loglog(E, F2)
    
    # Add some nice axes labels
    pl.xlabel(ares.util.labels['E'])
    pl.ylabel(ares.util.labels['flux'])    
    
Notice how the plot of ``F2`` has been hardened by neutral absorption in the IGM!
    
