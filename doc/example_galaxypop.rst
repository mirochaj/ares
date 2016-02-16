:orphan:

More Detailed Models: The GalaxyPopulation object
=================================================
Most of the global 21-cm examples in the documentation tie the volume-averaged emissivity of galaxies to the rate at which mass collapses into dark matter halos. Because of this, they are referred to as :math:`f_{\mathrm{coll}}` models throughout, and are selected by setting ``pop_model='fcoll'``. In the code, they are represented by ``GalaxyAggregate`` objects.

However, we can also run more detailed models in which the properties of galaxies are allowed to change as a function of halo mass, redshift, and/or potentially other quantities.

A few usual imports before we begin:

::

    import ares
    import numpy as np
    import matplotlib.pyplot as pl


A Simple GalaxyPopulation
-------------------------


::

    pars = \
    {
     'pop_Tmin': 1e4,
     'pop_model': 'sfe',
     'pop_Macc': 'mcbride2009',

     'pop_sed': 'leitherer1999',

     'pop_fesc': 0.2,

     'pop_fstar': 'php',
     'php_Mfun': 'dpl',
     'php_Mfun_par0': 0.15,
     'php_Mfun_par1': 1e12,
     'php_Mfun_par2': 0.5,
     'php_Mfun_par3': 0.5,

    }
    
::

    pop = ares.populations.GalaxyPopulation(**pars)
    
    MUV = np.linspace(-24, -10)
    
    
    pl.semilogy(MUV, pop.LuminosityFunction(4, MUV))
    
To compare to the observed galaxy luminosity function

::

    obslf = ares.analysis.ObservedLF()
    obslf.Plot(z=4, round_z=0.3)
    
The ``round_z`` makes it so that any dataset available in the range :math:`3.7 \leq z \leq 4.3`` gets included in the plot.






Parameterized Halo Properties (PHPs)
------------------------------------

::

    pars = \
    {
     'pop_Tmin': 1e4,
     'pop_model': 'sfe',
     'pop_Macc': 'mcbride2009',

     'pop_sed': 'leitherer1999',

     'pop_fstar': 'php[0]',
     'php_Mfun[0]': 'dpl',
     'php_Mfun_par0[0]': 0.15,
     'php_Mfun_par1[0]': 1e12,
     'php_Mfun_par2[0]': 0.5,
     'php_Mfun_par3[0]': 0.5,

     'pop_fesc': 'php[1]',
     'php_Mfun[1]': 'dpl',
     'php_Mfun_par0[1]': 0.2,
     'php_Mfun_par1[1]': 0.,
     'php_Mfun_par2[1]': 0.05,
     'php_Mfun_par3[1]': 0.,
     'php_Mfun_par4[1]': 1e8,

    }
    
::

    pop = ares.populations.GalaxyPopulation(**pars)
    
    MUV = np.linspace(-24, -10)
    
    pl.semilogy(MUV, pop.LuminosityFunction(4, MUV))


Compare to UV luminosity density of previous model.


Currently, the following parameters are supported by the PHP protocol:

* ``pop_fstar``
* ``pop_fesc``


   
    
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







