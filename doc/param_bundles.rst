:orphan:

Parameter Bundles
=================
The goal of ParameterBundles is to neatly package sets of commonly-used parameters with their most often-used values. This means you don't need to sift through the vast listing in `SetDefaultParameterValues` and attempt to determine which you'll need every time you run a new type of calculation. Instead, you can initialize a `ParameterBundle` object and make modifications rather than starting from scratch. Think of them as building blocks for a complete set of parameters.

This sort of functionality already exists to some degree given the different :doc:`problem_types` in *ares*. However, problem types are reserved for simulations only, whereas parameter bundles can be used to separately initialize the sub-components of a typical *ares* calculation, like `GalaxyPopulation` objects, parameters governing numerical approximations and the physics being included, etc.

In the future, the problem types in *ares* will probably be re-defined in terms of parameter bundles.

All bundles listed below can be created via, e.g., ::

    import ares
    
    pars = ares.util.ParameterBundle('pop:fcoll')

Populations
-----------
The following bundles return a base set of parameters that could be used to initialize a GalaxyPopulation object.

* ``pop:fcoll``
    A basic :math:`f_{\mathrm{coll}}`-based population described by the rate at which mass collapses onto dark matter halos exceeding some threshold mass (or equivalent virial temperature) and a constant star formation efficiency.
    
* ``pop:sfe`` or ``pop:lf``
    A population in which the star formation efficiency (SFE) is parameterized as a function of halo mass. This allows one to generate models for the galaxy luminosity function.
    
Spectral Energy Distributions
-----------------------------
The following bundles return a base set of parameters that can be added to the parameters of a Population bundle to modify its spectral energy distribution (SED). For example, ::

    pop_pars = ares.util.ParameterBundle('pop:fcoll')
    sed_pars = ares.util.ParameterBundle('sed:uv')

    pars = pop_pars + sed_pars
    
For calculations using multiple populations, you will need to give an identification number to each population via ::

    pars.num = 0
    
Currently, the following SED bundles are supported:    

* ``sed:uv``
    A simple SED in which the user sets the UV luminosity by hand.
    
* ``sed:pl``
    A simple power-law X-ray spectrum, which by default spans the energy range :math:`200 \leq h\nu/\mathrm{eV} \leq 30000`.

* ``sed:mcd``
    A multi-color disk black hole accretion spectrum. Assumes a :math:`10 \ M_{\odot}` BH.

* ``sed:bpass``
    A stellar SED from the *BPASS* code (version 1.0).

* ``sed:s99``
    A stellar SED from the original *starburst99* dataset.
    
Physics
-------

* ``physics:lwb``
    A few parameters that turn on a proper treatment of the Lyman-Werner background.
    

* ``physics:xrb``
    A few parameters that turn on a proper treatment of the X-ray background.
    
Simulations
-----------
If the bundle you specify is not defined in ares.util.ParameterBundles, *ares* will search for a module of the same name in ares/input/litdata. For more on these kinds of modules, see :doc:`uth_litdata`.

* ``mirocha2016:dpl``
    Parameters to initialize a simulation of the global 21-cm signal using a halo-mass-dependent star formation efficiency (a double power law (DPL) by default), as in `Mirocha, Furlanetto, & Sun (2017) <http://adsabs.harvard.edu/abs/2017MNRAS.464.1365M>`_. Changing the suffixed from ``dpl`` to ``steep`` or ``floor`` will instead use those models from the paper (see Figures 1 and 2). If you want to explore deviations from these models, check out the :doc:`params_populations` listing, especially the bit about parameterized halo properties.
    
    .. note :: For this to work "out of the box" you will need a lookup table for the IGM opacity that is not included with *ares* by default. See :doc:`inits_tables`: for more info on generating these lookup tables.

* ``mirocha2017:high``
    Parameters to augment the 'mirocha2016' simulations of the global 21-cm by adding in a simple prescription for PopIII stars. Changing the suffixed from ``high`` to ``low`` or ``med`` will assume different masses for PopIII stars. See :doc:`example_popIII` for more information.

Creating your own
-----------------
While some parameter bundles are defined in the source code (e.g., all but those in the ``Simulations'' section above), they can also be defined in separate files. For example, the ``mirocha2016:dpl`` model is defined in the ``dpl'' dictionary in the file ``mirocha2016.py'', which lives in ``$ARES/input/litdata''. You can write your own parameter bundles in the same way, just stick them in one of the following places (searched in this order!):

* ``$ARES/input/litdata''
* ``$HOME/.ares''
* Your current working directory.



