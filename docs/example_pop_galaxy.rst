:orphan:

More Realistic Galaxy Populations
=================================
Most global 21-cm examples in the documentation tie the volume-averaged emissivity of galaxies to the rate at which mass collapses into dark matter halos (this is the default option in *ARES*). Because of this, they are referred to as :math:`f_{\text{coll}}` models throughout, and are selected by setting ``pop_sfr_model='fcoll'``. In the code, they are represented by ``GalaxyAggregate`` objects, named as such because galaxies are only modeled in aggregate, i.e., there is no distinction in the properties of galaxies as a function of mass, luminosity, etc.

However, we can also run more detailed models in which the properties of galaxies are allowed to change as a function of halo mass, redshift, and/or potentially other quantities.

A few usual imports before we begin:

::

    import ares
    import numpy as np
    import matplotlib.pyplot as pl


A Simple Galaxy Population
---------------------------
The most common extension to simple models is to allow the star formation efficiency (SFE) to vary as a function of halo mass. This is motivated observationally by the mismatch in the shape of the galaxy luminosity function (LF) and dark matter halo mass function (HMF). In `Mirocha, Furlanetto, & Sun (2017) <http://adsabs.harvard.edu/abs/2017MNRAS.464.1365M>`_, we adopted a double power-law form for the SFE, i.e., 

.. math::

    f_{\ast}(M_h) = \frac{2 f_{\ast,p}} {\left(\frac{M_h}{M_{\text{p}}} \right)^{\gamma_{\text{lo}}} + \left(\frac{M_h}{M_{\text{p}}}  \right)^{\gamma_{\text{hi}}}}

where the free parameters are the normalization, :math:`f_{\ast,p}`, the peak mass, :math:`M_p`, and the power-law indices in the low-mass and high-mass limits, :math:`\gamma_{\text{lo}}` and :math:`\gamma_{\text{hi}}`, respectively. Combined with a model for the mass accretion rate onto dark matter halos (:math:`\dot{M}_h`; see next section), the star formation rate as computed as

.. math::

    \dot{M}_{\ast} = f_{\ast} \left(\frac{\Omega_{b,0}}{\Omega_{m,0}} \right) \dot{M}_h
    
In general, the SFE curve must be calibrated to an observational dataset (see :doc:`example_mcmc_lf`), but you can also just grab our best-fitting parameters for a redshift-independent SFE curve as follows:

::

    p = ares.util.ParameterBundle('mirocha2017:base')
    pars = p.pars_by_pop(0, strip_id=True)
    
The second command extracts only the parameters associated with population #0, which is the stellar population in this calculation (population #1 is responsible for X-ray emission only; see :doc:`example_gs_multipop` for more info on the approach to populations in *ARES*). Passing ``strip_id=True`` removes all ID numbers from parameter names, e.g., ``pop_sfr_model{0}`` becomes ``pop_sfr_model``. The reason for doing that is so we can generate a single ``GalaxyPopulation`` instance, e.g.,

::

    pop = ares.populations.GalaxyPopulation(**pars)
    
If you glance at the contents of ``pars``, you'll notice that the parameters that define the double power-law share a ``pq`` prefix. This is short for "parameterized quantity", and is discussed more generally on the page about :doc:`uth_pq`.

.. note::
    You can access population objects used in a simulation via the ``pops`` attribute, which is a list of population objects that belongs to instances of  common simulation classes like ``Global21cm``, ``MetaGalacticBackground``, etc.


Now, to generate a model for the luminosity function, simply define your redshift of interest and array of magnitudes (assumed to be rest-frame :math:`1600 \unicode{x212B}` AB magnitudes), and pass them to the aptly named ``LuminosityFunction`` function,

::

    z = 6
    MUV = np.linspace(-24, -10)
    lf = pop.LuminosityFunction(z, MUV)
    
    pl.figure(1)
    pl.semilogy(MUV, lf)
    
To compare to the observed galaxy luminosity function, we can use some convenience routines setup to easily access and plot measurements stored in the *ARES* ``litdata`` module:

::

    obslf = ares.analysis.GalaxyPopulation()
    obslf.Plot(z=z, round_z=0.2)
    pl.ylim(1e-8, 10)
    pl.legend()
    
    pl.savefig('ares_pop_galaxy_lf6.png')
    
.. figure::  https://www.dropbox.com/s/9zcrikk2ptrb6dz/ares_pop_galaxy_lf6.png?raw=1
   :align:   center
   :width:   600

   Simple galaxy evolution model with :math:`M_h`-dependent SFE compared to UV luminosity functions at :math:`z\sim 6`. Model calibrated only to the Bouwens et al. (2015) points.
    
    
The ``round_z`` makes it so that any dataset available in the range :math:`3.7 \leq z \leq 4.3`` gets included in the plot. To do this for multiple redshifts at the same time, you could do something like:

::

    redshifts = [5,6,7,8]
    MUV = np.linspace(-24, -10)

    # Create a 1x4 panel plot, include all available data sources
    mp = obslf.MultiPlot(redshifts, round_z=0.3, ncols=4, sources='all', fig=2, mp_kwargs=dict(padding=(0.2,0.2)))
    
    for i, z in enumerate(redshifts):

        obslf.Plot(z=z, round_z=0.3, ax=mp.grid[i])
        
        lf = pop.LuminosityFunction(z, MUV)

        mp.grid[i].semilogy(MUV, lf)
    
    obslf.add_master_legend(mp, ncol=3)
    
    pl.figure(2)
    pl.savefig('ares_pop_galaxy_lf_allz.png')


.. figure::  https://www.dropbox.com/s/2g3mf2s7beeuuwj/ares_pop_galaxy_lf_allz.png?raw=1
   :align:   center
   :width:   1200

   Simple galaxy evolution model with :math:`M_h`-dependent SFE compared to UV luminosity functions at :math:`5 \lesssim z \lesssim 8`. Again, model calibrated only to the Bouwens et al. (2015) points at :math:`z \sim 6`.
    

To create the ``GalaxyPopulation`` used above from scratch (i.e., without using parameter bundles), we could have just done:

::

    pars = \
    {
     'pop_sfr_model': 'sfe-func',
     'pop_sed': 'eldridge2009',

     'pop_fstar': 'pq',
     'pq_func': 'dpl',
     'pq_func_par0': 0.05,
     'pq_func_par1': 2.8e11,
     'pq_func_par2': 0.51,
     'pq_func_par3': -0.61,
     'pq_func_par4': 1e10,  # halo mass at which SFE is normalized
    }
    
    pop = ares.populations.GalaxyPopulation(**pars)
    
	
.. note :: Beware that by default, the double power-law is normalized at :math:`M_h = 10^{10} \ M_{\odot}` (see ``ps_func_par4`` above), whereas the Equation above for :math:`f_{\ast}` is defined such that ``pq_func_par0`` refers to the SFE at the peak mass. If you prefer a peak-normalized SFE, you can set ``pq_func='dpl_normP'`` instead.
    
Accretion Models
~~~~~~~~~~~~~~~~
By default, *ARES* will derive the mass accretion rate (MAR) onto halos from the HMF itself (see Section 2.2 of `Furlanetto et al. 2017 <http://adsabs.harvard.edu/abs/2017MNRAS.472.1576F>`_. for details). That is, ``pop_MAR='hmf'`` by default. There are also two other options:

* Plug-in your favorite mass accretion model as a lambda function, e.g., ``pop_MAR=lambda z, M: 1. * (M / 1e12)**1.1 * (1. + z)**2.5``.
* Grab a model from ``litdata``. The median MAR from McBride et al. (2009) is included (same as above equation), and can used as ``pop_MAR='mcbride2009'``. If you'd like to add more options, use ``$ARES/input/litdata/mcbride2009.py`` as a guide.

.. warning:: Note that the MAR formulae determined from numerical simulations may not have been calibrated at the redshifts most often targeted in *ARES* calculations, nor are they guaranteed to be self-consistent with the HMF used in *ARES*. One approach used in `Sun \& Furlanetto (2016) <http://adsabs.harvard.edu/abs/2016MNRAS.460..417S>`_ is to re-normalize the MAR by requiring its integral to match that predicted by :math:`f_{\text{coll}}(z)`, which can boost the accretion rate at high redshifts by a factor of few. Setting ``pop_MAR_conserve_norm=True`` will enforce this condition in *ARES*.

See :doc:`uth_pop_halo` for more information.

   
Dust
~~~~
Correcting for reddening due to the presence of dust in star-forming galaxies can be extremely important, especially in massive galaxies. When calling upon the ``LuminosityFunction`` method as in the above example, be aware that **all magnitudes are assumed to be observed magnitudes, not intrinsic magnitudes.** 

At its simplest, the dust correction looks as follows (e.g., `Meurer et al. 1999) <https://ui.adsabs.harvard.edu/abs/1999ApJ...521...64M/abstract>`_

.. math::

    A_{\text{UV}} = a + b \beta
    
where :math:`\beta` is the rest-frame UV slope, and :math:`a` and :math:`b` are empirically-derived constants. 

Some common dust corrections can be accessed by name and passed in via the ``dustcorr_method`` parameter:

* ``meurer1999``
* ``pettini1998``

By default, *ARES* will assume a constant :math:`\beta=-2`. However, in general this is a poor approximation: fainter galaxies are known to suffer less from dust reddening than bright galaxies. Simply set ``dustcorr_beta='bouwens2014'``, for example, to adopt the Bouwens et al. 2014 :math:`M_{\text{UV}}-\beta` relation.

To employ the self-consistent model put forth in `Mirocha, Mason, & Stark (2020) <https://ui.adsabs.harvard.edu/abs/2020arXiv200507208M/abstract>`_, see :doc:`example_pop_dusty`.



