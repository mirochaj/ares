:orphan:

Parameter Bundles
=================
The goal of ParameterBundles is to neatly package sets of commonly-used parameters with their most often-used values. This means you don't need to sift through the vast listing in `SetDefaultParameterValues` and attempt to determine which you'll need every time you run a new type of calculation. Instead, you can initialize a `ParameterBundle` object and make modifications from there.

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

Other
-----
If the bundle you specify is not defined in ares.util.ParameterBundles, *ares* will search for a module of the same name in ares/input/litdata. For more on these kinds of modules, see :doc:`uth_litdata`.

* ``mirocha2016``
    Initialize a simulation of the global 21-cm signal using the galaxy
    luminosity function as the underlying model, as in `Mirocha, Furlanetto, \& Sun (2016) <http://arxiv.org/abs/1607.00386>`_. Note that for this to work "out of the box," you will need a few lookup tables...



