:orphan:

Dark Matter Halo Populations
============================
Most of our models for star formation, black hole formation, etc., are painted onto a population of dark matter (DM) halos. In the simplest models, we just care about the total fraction of matter residing in halos, or the rate of change in that quantity (e.g., ``fcoll`` :doc:`uth_pop_sfrd`). In slightly more sophisticated models, we'll need the halo mass function, which describes the number density of halos as a function of redshift and mass, and perhaps the growth rates of halos (again, with redshift and mass).

For all of this, *ares* uses the `hmf <http://hmf.readthedocs.org/en/latest/>`_ code. In order to speed-up calculations, by default, *ares* will read-in a lookup table for the mass function rather than using *hmf* to generate mass functions on-the-fly. This saves *a lot* of time. 

    .. note :: You do not necessarily need to install `hmf` to use *ares*, e.g., if the default lookup table is OK for your purposes. However, you should know that this table was generated with `hmf`, and reference the `relevant paper <https://arxiv.org/abs/1306.6721>`_ in your work.

To initialize a halo population on its own (i.e., without any information about sources that live in the halos), do

::
    
    import ares
    
    pop = ares.populations.HaloPopulation()
    
This class is inherited by almost every other kind of source population available in *ares*. Its most important attribute is simply called `halos`, and itself is an instance of the ``HaloMassFunction`` class, which does the heavy lifting. The attributes you're most likely to need access to are:

+ ``'tab_z'``
    The array of redshifts over which we have tabulated the HMF.
+ ``'tab_M'``
    The array of masses over which we have tabulated the HMF.    
+ ``'tab_dndm'``
    The differential halo mass function, i.e., number of halos per mass bin, :math:`dn/dm`. Note that the shape should be ``(len(z), ``len(M))``.
+ ``'tab_fcoll'``
    Fraction of matter in collapsed halos as a function of redshift and lower limit of integration (see below).
    
To have a look at the mass function at a few redshifts, you could do something like:

::

    import numpy as np
    import matplotlib.pyplot as pl
    
    for z in [4, 6, 10]:
        i = np.argmin(np.abs(z - pop.halos.tab_z))
        
        pl.loglog(pop.halos.tab_M, pop.halos.tab_dndm[i,:])
        
    # Tighten up, since high-mass end will stretch out y-axis a lot    
    pl.ylim(1e-25, 10)
        
.. note :: The default lookup table only spans the range :math:`3 \leq z \leq 60`, and :math:`4 \leq log_{10} M \leq 16`.
    
The Collapsed Fraction
~~~~~~~~~~~~~~~~~~~~~~
Because it is used in simple models for star formation at high-z, the fraction of mass in collapsed DM halos (above some threshold mass) is pre-computed as a function of redshift and minimum mass, and stored in the default lookup table. That is, we have at our disposal

.. math :: f_{\mathrm{coll}}(m > m_{\min},z) = \rho_m^{-1} \int_{M_{\min}}^{\infty} m \frac{dn}{dm} dm
    
where :math:`m` is the halo mass, :math:`\rho_m` is the mean matter density today, and :math:`dn/dm` is the differential mass function.
    
.. note :: We can use this table to compute the fraction of mass in a finite mass range simply by subtracting off :math:`f_{\mathrm{coll}}(M_{\max},z)`.
    
For a quick sanity check, you could re-derive :math:`f_{\mathrm{coll}}` from the mass function:

::
    
    # Arbitrarily choose a minimum mass of 10^8 Msun
    i = np.argmin(np.abs(pop.halos.tab_M - 1e8))
    
    pl.semilogy(pop.halos.tab_z, pop.halos.tab_fcoll[:,i])
    
    # Compute it ourselves
    integrand = pop.halos.tab_M[i:] * pop.halos.tab_dndm[:,i:]
    fcoll = np.trapz(integrand, x=pop.halos.tab_M[i:], axis=1) / pop.cosm.mean_density0
    pl.semilogy(pop.halos.tab_z, fcoll, ls='--', lw=3)

Notice that we carry around the mean matter density at :math:`z=0` in an instance of the Cosmology class, which hangs off of the population object in the ``cosm`` attribute. It has units of :math:`M_{\odot} \ \mathrm{cMpc}^{-3}`, so we did not need to do any unit conversions.

There are also some built-in routines to compute :math:`f_{\mathrm{coll}}` and its derivatives at arbitrary redshifts, see attributes ``fcoll``, ``dfcolldz``, and ``dfcolldt``.

Halo Growth Rates
~~~~~~~~~~~~~~~~~
For some models we need to know the growth rates of halos, in addition to their space density. There are a few ways to go about this.

The default option in *ares* is to use the mass function itself to derive halo mass accretion rates, as is discussed in Section 2.2 of `Furlanetto et al. 2017 <http://adsabs.harvard.edu/abs/2017MNRAS.472.1576F>`_. This approach assumes that halos evolve at fixed number density, which of course is not true in detail, but it is ultimately useful nonetheless as it preserves self-consistency between the abundance of halos and their growth histories.

To plot the growth rates, you can do, e.g.,

::

    M = np.logspace(9, 13)
    for z in [4, 6, 10]:
        pl.loglog(M, pop.MGR(z, M))


Alternatively, you can supply your own function for the mass growth rates, perhaps those from simulations. For example, we could use the median mass accretion rate found by McBride et al. 2009, 

::

    MAR = lambda z, Mh: 24.1 * (Mh / 1e12)**1.094 * (1. + 1.75 * z) * (1. + z)**1.5
    
    pop = ares.populations.HaloPopulation(pop_MAR=MAR)
    
and compare to our previous plot,

::

    M = np.logspace(9, 13)
    for z in [4, 6, 10]:
        pl.loglog(M, pop.MGR(z, M), ls='--')
        
The agreement is decent considering the simplicity of the default model. Plus, few simulations have attempted to calibrate this relationship at high redshifts.         

Generating new HMF Tables
~~~~~~~~~~~~~~~~~~~~~~~~~
If the default lookup table doesn't suit your purpose, you can (i) generate your own using the same machinery, or (ii) create your own lookup table using some other code. 

If all you want to do is change the redshift or mass ranges, resolution, cosmological parameters, or model for the mass function (e.g., Press-Schechter, Sheth-Tormen, etc.), I'd recommend option \#1. If you navigate to ``$ARES/input/hmf``, you can modify the script ``generate_hmf_tables.py``. Have a look at :doc:`params_hmf` to see what changes are possible. By default, *ares* will go looking in ``$ARES/input/hmf`` for suitable lookup tables, so your new table will be found automatically if you supply the same set of parameters to an *ares* simulation. If you want to make these changes permanent without modifying the source code locally, you could change your custom defaults (see :doc:`params` for instructions).

If you have your own code for generating the halo mass function, everything else in *ares* should work as-advertised so long as the format of your table matches the expected format. Right now, *ares* supports pickle files ``.npy`` or ``.npz`` files, and HDF5 files. Have a look in ``ares.physics.HaloMassFunction.save`` to see the expected order and/or names of fields in your file. Once you've got a complete file, you'll want to provide the full path to *ares* via the ``hmf_table`` parameter.







