:orphan:

Parameterized Models for Sources and Their Spectra
--------------------------------------------------
If you want to use *ares*'s numerical solvers, but don't care for the
available slew of input options (e.g., spectra, luminosity functions, etc.),
you can pass in functions of your own. 

There are a few magical override parameters to help you with this:

* `pop_spectrum`: a Python function of a single variable (photon energy)
* `pop_rhoL`: a Python function of a single variable (redshift)

You can either supply Python functions for these parameters yourself, or use results from the literature. We'll demonstrate each approach below.

As always, start with a few imports:

::

    import ares
    import numpy as np
    import matplotlib.pyplot as pl


User-Defined Models
-------------------
Create Python functions.



Using Constraints from the Literature
-------------------------------------
Some SED and luminosity density models will definitely be used over and over again. This motivated the creation of the ``litdata'' module, which you can use on its own as a convenient way of reading in models from the literature (see :doc:`uth_litdata`) or through more sophisticated calculations, e.g., :doc:`example_crb_uv`. If one of your user-defined models is common in the literature, consider forking *ares* and adding it to the database! See :doc:`uth_litdata` for instructions on how to do that, or simply emulate pre-existing modules in ``$ARES/input/litdata``.

For example, to use the template quasar spectrum from `Sazonov et al. (2004) <http://adsabs.harvard.edu/abs/2004MNRAS.347..144S>`_ and the luminosity density evolution of `Ueda et al. (2003) <http://adsabs.harvard.edu/abs/2003ApJ...598..886U>`_, you would modify the parameters from the :doc:`example_crb_uv` example as follows:

::
    
    pars = \
    {
     'pop_type': 'galaxy',
     'pop_sed': 'sazonov2004',            # NEW
     'pop_rhoL': 'ueda2003',              # NEW
     'pop_kwargs': {'evolution': 'ple'},  # NEW

     'pop_Emin': 1.0,
     'pop_Emax': 5e4,
     'pop_EminNorm': 2e3,
     'pop_EmaxNorm': 1e4,

     # Solution method
     'pop_solve_rte': True,
     'pop_tau_Nz': 100,
     'include_H_Lya': False,

     'sawtooth_nmax': 8,

     'initial_redshift': 6,
     'final_redshift': 3,
    }

Notice the aforementioned ``pop_sed`` and ``pop_rhoL`` parameters, in addition to ``pop_kwargs``, which we have yet to mention. The contents of ``pop_kwargs`` will be passed to all functions in associated with the ``pop_rhoL`` module, since there are many options to be specified, e.g.:

* Evolution in the luminosity function, in this case, a ''pure luminosity evolution'' (``'ple'``) model. The other options from Ueda et al. (2003) include pure density evolution (``'pde'``) and luminosity-dependent density evolution (``'ldde'``).

Now, initialize a simulation instance and run it in the usual way:

::

    rad = ares.simulations.MetaGalacticBackground(pop_sawtooth=True, 
        approx_tau=None, **pars)

    # Compute background flux
    rad.run()
    
Instead of extracting fluxes using ``rad.get_history``, as in the :doc:`example_crb_lw` and :doc:`example_crb_uv` examples, we'll use some built-in analysis routines to plot the background intensity at :math:`z=3`:

::

    anl = ares.analysis.MetaGalacticBackground(rad)
    ax = anl.PlotBackground(z=3, color='b')

For a sanity check, let's use yet another ``litdata`` submodule, containing the meta-galactic background as computed in `Haardt & Madau (2012) <http://adsabs.harvard.edu/abs/2012ApJ...746..125H>`_ (their ''galaxies+quasars'' model):

::

    hm12 = ares.util.read_lit('haardt2012')
    z, E, flux = hm12.MetaGalacticBackground()
    
    # Find element closest to z=3
    j = np.argmin(np.abs(z - 3.))
    
    # Plot it on pre-existing axes!
    ax.plot(E, flux[j] / 1e-21, color='c')

    ax.set_xlim(1, 4e3)
    pl.draw()

This is not even close to an apples-to-apples comparison, but let's worry about that more later.






