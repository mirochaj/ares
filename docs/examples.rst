Examples
========

Running Individual Simulations for Reionization and 21-cm
---------------------------------------------------------
These examples show to run 21-cm calculations, which also contain the mean reionization and thermal histories. Here, we focus on relatively simple source populations to start, but the idea is that one can swap in more complicated models (as discussed in the next section) easily.

.. toctree::
    :maxdepth: 1

    examples/example_gs_standard
    examples/example_gs_multipop

Advanced Source Populations
---------------------------
These examples show how to work with source populations individually, i.e., not as part of a larger simulation. So, if you're just interested in, e.g., modeling galaxy luminosity functions, or using a more sophisticated galaxy model for 21-cm calculations, this should be a good starting point.

.. toctree::
    :maxdepth: 2

    examples/example_pop_galaxy
    examples/example_galaxies_demo
    examples/example_pop_popIII
    examples/example_pop_dusty
    * :doc:`example_edges`

Parameter Studies and Inference
-------------------------------
As of version 1.0, ARES does not contain any wrappers around MCMC samplers or routines to help facilitate MCMC analysis. The rationale for this decision was that each particular problem is sufficiently different that one usually needs to customize the fitting procedure anyways. So, the following examples hopefully give a good impression of what this looks like with ARES, but ARES is really just being used as a callable model here. If anybody would like to write up examples for samplers in addition to emcee, that would be great!

.. toctree::
    :maxdepth: 1

    example_ham
    example_mcmc_gs
    example_mcmc_lf


The Meta-Galactic Radiation Background
--------------------------------------

.. toctree::
    :maxdepth: 1

    examples/example_crb_uv
    examples/example_crb_xr

1-D Radiative Transfer
----------------------
Maybe nobody is using this anymore, but ARES can do radiative transfer in 1-D! It's actually how the code began, back in ~2011, when it was called `rt1d`. Contact me if you have problems with this stuff, it has been collecting dust for some time.

.. toctree::
    :maxdepth: 1

    example_rt06_1
    example_rt06_2
    example_adv_RT_w_He

Extensions
----------
.. toctree::
    :maxdepth: 2

    examples/example_litdata
    example_embed_ares
    examples/uth_pq
    uth_pop_new
