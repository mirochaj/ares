:orphan:

Parameter Bundles
=================
The goal of ParameterBundles is to neatly package sets of commonly-used parameters with their most often-used values. This means you don't need to sift through the vast listing in `SetDefaultParameterValues` and attempt to determine which you'll need every time you run a new type of calculation. Instead, you can initialize a `ParameterBundle` object, and make modifications from there.

This sort of functionality already exists to some degree given the different :doc:`problem_types` in *ares*. However, problem types are reserved for simulations only, whereas parameter bundles can be used to separately initialize the sub-components of a typical *ares* calculation, like `GalaxyPopulation` objects, parameters governing numerical approximations and the physics being included, etc.

In the future, the problem types in *ares* will probably be re-defined in terms of parameter bundles.

Sources
-------





Simulations
-----------

``gs``
    Initialize a simple model for the global 21-cm signal.
    
``rt1d``
    Initialize a 1-D radiative transfer calculation.

.. note :: If the bundle you specify is not defined in the appropriate object 
    in ares.util.ParameterBundles, *ares* will search for a module of the same
    name in ares/input/litdata. For more on these kinds of modules, see 
    :doc:`uth_litdata`.

``mirocha2016``
    Initialize a simulation of the global 21-cm signal using the galaxy
    luminosity function as the underlying model, as in (insert citation here).  
