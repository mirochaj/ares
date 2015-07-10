The Metagalactic X-ray Background
=================================
In this example, we'll compute the Meta-Galactic X-ray background over a
series of redshifts at high redshifts (:math:`10 \leq z \leq 40):

::
    
    # Initialize radiation background
    pars = \
    {
     # Source properties
     'pop_type': 'galaxy',
     'pop_sfrd': lambda z: 0.1,
     'pop_sed': 'pl',
     'pop_alpha': -1.5,
     'pop_Emin': 1e2,
     'pop_Emax': 3e4,
     'pop_EminNorm': 5e2,
     'pop_EmaxNorm': 8e3,
     'pop_yield': 2.6e39,
     'pop_yield_units': 'erg/s/sfr',

     'initial_redshift': 40.,
     'final_redshift': 10.,
    }
    
To summarize these inputs, we've got:

* A constant SFRD of :math:`0.1 \ M_{\odot} \ \mathrm{yr}^{-1} \ \mathrm{cMpc}^{-3}`, given by the ``pop_sfrd`` parameter.
* A power-law spectrum with index :math:`\alpha=-1.5`, given by ``pop_sed`` and ``pop_alpha``, extending from 0.1 keV to 30 keV.
* A yield of :math:`2.6 \times 10^{39} \ \mathrm{erg} \ \mathrm{s}^{-1} \ (M_{\odot} \ \mathrm{yr})^{-1}` in the :math:`0.5 \leq h\nu / \mathrm{keV} \leq  8` band, set by ``pop_EminNorm``, ``pop_EmaxNorm``, ``pop_yield``, and ``pop_yield_units``. This is the :math:`L_X-\mathrm{SFR}` relation found by `Mineo et al. (2012) <http://adsabs.harvard.edu/abs/2012MNRAS.419.2095M>`_.

See :doc:`params_populations` for a complete listing of parameters relevant to :class:`ares.populations.GalaxyPopulation` objects.
    
Now, to initialize a calculation:

::  

    import ares

    mgb = ares.simulations.MetaGalacticBackground(**pars)
    
So long as ``verbose=True`` (which it is by default), you should see the following output to the screen:

::

    ##########################################################################
    ####                        Galaxy Population                         ####
    ##########################################################################
    #### ---------------------------------------------------------------- ####
    #### Redshift Evolution                                               ####
    #### ---------------------------------------------------------------- ####
    #### SFRD        : parameterized                                      ####
    #### ---------------------------------------------------------------- ####
    #### Radiative Output                                                 ####
    #### ---------------------------------------------------------------- ####
    #### yield (erg / s / SFR) : 2.6e+39                                  ####
    #### EminNorm (eV)         : 500                                      ####
    #### EmaxNorm (eV)         : 8000                                     ####
    #### ---------------------------------------------------------------- ####
    #### Spectrum                                                         ####
    #### ---------------------------------------------------------------- ####
    #### SED               : pl                                           ####
    #### Emin (eV)         : 200                                          ####
    #### Emax (eV)         : 30000                                        ####
    #### alpha             : 1                                            ####
    #### logN              : -inf                                         ####
    ##########################################################################

Again, as in the previous examples, this is really just to provide a sanity check.

Now, let's run the thing:

::

    mgb.run()
    
We'll pull out the evolution of the background just as we did in the previous two examples:

::

    z, E, flux = mgb.get_history(flatten=True)

and plot up the result:

::

    from ares.physics.Constants import erg_per_ev

    pl.semilogy(E, flux[-1] * E * erg_per_ev, color='k')
    pl.xlabel(ares.util.labels['E'])
    pl.ylabel(ares.util.labels['flux_E'])
    
    z, E, flux = mgb.get_history(flatten=True)
                
Compare to the analytic solution, given by Equation A1 in `Mirocha (2014) <http://adsabs.harvard.edu/abs/2014arXiv1406.4120M>`_ (the *cosmologically-limited* solution to the radiative transfer equation)

.. math ::

    J_{\nu}(z) = \frac{c}{4\pi} \frac{\epsilon_{\nu}(z)}{H(z)} \frac{(1 + z)^{9/2-(\alpha + \beta)}}{\alpha+\beta-3/2} \times \left[(1 + z_i)^{\alpha+\beta-3/2} - (1 + z)^{\alpha+\beta-3/2}\right]

with :math:`\alpha = -1.5`, :math:`\beta = 0`, :math:`z=10`, and :math:`z_i=40`,

::

    # Grab the GalaxyPopulation instance
    pop = mgb.pops[0] 

    # Compute cosmologically-limited solution
    e_nu = np.array(map(lambda E: pop.Emissivity(10., E), E))
    e_nu *= c / 4. / np.pi / pop.cosm.HubbleParameter(10.) 
    e_nu *= (1. + 10.)**6. / -3.
    e_nu *= ((1. + 40.)**-3. - (1. + 10.)**-3.)
    e_nu *= ev_per_hz

    # Plot it
    pl.semilogy(E, e_nu, color='k', ls='-')
    
Neutral Absorption by the Diffuse IGM
-------------------------------------   
The calculation above is basically identical to the optically-thin LW and UV background calculations performed in the previous two examples, at least in the cases where we neglected any sawtooth effects. While there is no modification to the X-ray background due to resonant absorption in the Lyman series (of Hydrogen or Helium II), bound-free absorption by intergalactic hydrogen and helium atoms acts to harden the spectrum. By default, *ares* will not include these effects.

To "turn on" bound-free absorption in the IGM, modify the dictionary of parameters you've got already:

::

    pars['approx_tau'] = 'neutral'

Now, initialize and run a new calculation:

::

    mgb2 = ares.simulations.MetaGalacticBackground(**pars)
    mgb2.run()
    
and plot the result on the same axes:

::

    z2, E2, flux2 = mgb2.get_history(flatten=True)

    pl.loglog(E2, flux2[-1] * E2 * erg_per_ev, color='k', ls=':')
    
The behavior at low photon energies (:math:`h\nu \lesssim 0.3 \ \mathrm{keV}`)
is an artifact that arises due to poor redshift resolution. This is a trade
made for speed in solving the cosmological radiative transfer equation,
discussed in detail in Section 3 of `Mirocha (2014)
<http://adsabs.harvard.edu/abs/2014arXiv1406.4120M>`_. For more accurate
calculations, you must enhance the redshift sampling using the ``pop_tau_Nz``
parameter, e.g.,

::

    pars['pop_tau_Nz'] = 500

The optical depth lookup tables that ship with *ares* use ``pop_tau_Nz=400``
as a default. If you run with ``pop_tau_Nz=500``, you should see some improvement in the soft X-ray spectrum. It'll take a few minutes to generate a new table. Run `$ARES/input/optical_depth/generate_optical_depth_tables.py` to make more!

.. note :: Development of a dynamic optical depth calculation is underway, which can be turned on and off using the ``dynamic_tau`` parameter.
    
Tabulating the Optical Depth    
----------------------------
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


Alternative Methods
-------------------
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
    
Self-Consistent Meta-Galactic Background & IGM
----------------------------------------------
If we don't already know the IGM optical depth *a-priori*, then the calculations above will only bracket the result expected in a more complex, evolving IGM, in which the radiation background ionizes the IGM, thus making the IGM more transparent, which then softens the meta-galactic background, and so on. To treat this interplay carefully, we need to...

