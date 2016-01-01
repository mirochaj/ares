:orphan:

Alternatives to fcoll models
============================
Most of the global 21-cm examples in the documentation tie the volume-averaged emissivity of galaxies to the rate at which mass collapses into dark matter halos. Because of this, they are referred to as :math:`f_{\mathrm{coll}}` models throughout, and are selected by setting ``pop_model='fcoll'``.

Here are a few more options for source populations.

A few imports before we begin:

::

    import ares
    import numpy as np
    import matplotlib.pyplot as pl


Halo Abundance Matching (HAM)
-----------------------------
This technique relates measurements of the galaxy luminosity function (LF) to the dark matter halo mass function by assuming galaxies of luminosity :math:`L` with number density :math:`\phi(L)` live in halos of mass :math:`M_h`, whose number density is given by the HMF, :math:`n(M_h)`. By enforcing the condition:

.. math::

    \int_{L(M_h)} \phi(L) dL = \int_{M_h} n(M_h) dM_h
    
we can solve for the mass-to-light relationship, :math:`L(M_h)`. This allows us to model the galaxy population at lower luminosities and higher redshifts than have so far been observed, provided some model for extrapolating :math:`L(M_h)` in mass and redshift.

The two most critical ingredients in this model are constraints on the luminosity function, and a model for the HMF. Let's grab the constraints from Bouwens et al. (2015) to start:

::

    b15 = ares.util.read_lit('bouwens2015')

The object ``b15`` has four important attributes: 

    - ``info``
    - ``redshifts``
    - ``data``
    - ``fits``

The first gives a full reference to the paper, while the ``data`` and ``fits`` entries tell us where in the paper the data is located.

For example, the best-fit Schecter parameters are:

::    

    print b15.fits['lf']['pars']
    
    
with corresponding redshifts:

::

    print b15.redshifts

    
.. note:: Check out the primer on :doc:`uth_litdata` if you 
    haven't already!
    
To initialize a population, we must importantly set ``pop_model='ham'``:

::

    pars = \
    {
     'pop_model': 'ham',
     'pop_Macc': 'mcbride2009',         # Halo MAR
     'pop_constraints': 'bouwens2015',  # Galaxy LF
     'pop_kappa_UV': 1.15e-28,          # Luminosity / SFR
    }

    pop = ares.populations.GalaxyPopulation(**pars)

This population instance has an attribute ``ham``, which carries along all of the abundance matching info. For starters, let's just plot up the constraints on the LF at each redshift, which have been converted from magnitudes to rest-frame :math:`1500\AA` luminosities:

::

    L = np.logspace(27., 30.)
    for i, z in enumerate(pop.ham.constraints['z']):
        Lh, phi = pop.ham.LuminosityFunction(z)
        pl.loglog(Lh, phi, ls='--', lw=3)

    pl.xlabel(r'$L \ (\mathrm{erg} \ \mathrm{s}^{-1} \ \mathrm{Hz}^{-1})$')
    pl.ylabel(r'$\phi(L)$')
    pl.xlim(1e27, 1e30)
    pl.ylim(1e-39, 1e-29)
    pl.show()

Now, the key quantity yielded by the abundance matching procedure is $L_h(M_h)$, which we convert to a star-formation efficiency (SFE), :math:`f_{ast}`, assuming a model for the halo mass accretion rate (from McBride et al. (2009); see ``pop_Macc`` parameter above):

::

    colors = ['r', 'b', 'g', 'k', 'm']
    for i, z in enumerate(pop.ham.redshifts):
        j = pop.ham.redshifts.index(z)
        Mh = pop.ham.MofL_tab[j]
        pl.scatter(Mh, pop.ham.fstar_tab[j], label=r'$z=%g$' % z, color=colors[i], marker='o', facecolors='none')

    pl.plot([1e8, 1e15], [0.2]*2, color='k', ls=':')
    pl.plot([1e8, 1e15], [0.3]*2, color='k', ls=':')
    pl.xlabel(r'$M_h / M_{\odot}$')
    pl.ylabel(r'$f_{\ast}$')
    pl.legend(ncol=1, frameon=False, fontsize=16, loc='lower right')

    Marr = np.logspace(8, 14)
    for i, z in enumerate(pop.ham.redshifts):
        j = pop.ham.redshifts.index(z)

        fast = pop.ham.SFE(z=z, M=Marr)
        pl.loglog(Marr, fast, color=colors[i])

You can also access the SFRD via ``pop.ham.SFRD``, which just integrates the product of the SFE and MAR over the mass function.

.. note:: You can run simulations of the global 21-cm using the HAM model for 
    source populations. Just be sure to pass in the appropriate parameters, as 
    ``pop_model != 'ham'`` by default!

    
Extrapolation options
~~~~~~~~~~~~~~~~~~~~~
In the above example defaults were used to extrapolate the SFE to low masses and high redshifts. There are several options for this, which are listed below, which should be supplied via the ``pop_ham_Mfun`` and ``pop_ham_zfun`` parameters as strings.

+------------+------------+----------------------------------+
| Dimension  |    :math:`f_{\ast}(M,z)` options              |
+============+============+===================+==============+
| logM       |  ``poly``  |  ``lognormal``    |              |
+------------+------------+-------------------+--------------+
| (1+z)      |  ``poly``  |  ``linear_t``     | ``constant`` |
+------------+------------+-------------------+--------------+


+------------+------------+-------------------+--------------+
| Dimension  |    :math:`L_h(M_h)` options                   |
+============+============+===================+==============+
| logM       |  ``poly``  |  ``pl``           |              |
+------------+------------+-------------------+--------------+
| (1+z)      |  ``poly``  |  ``linear_t``     | ``constant`` |
+------------+------------+-------------------+--------------+



Halo Occupation Distribution (HOD)
----------------------------------
Not yet implemented.






