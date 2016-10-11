:orphan:

Models for Radiation Emitted by Galaxies
========================================
There are three main ways to model the radiation emitted by galaxies, governed largely by whether or not ``pop_sed_model`` is ``True`` or  ``False``. 

``pop_sed_model=True``
~~~~~~~~~~~~~~~~~~~~~~
In this case, we're assuming that the source population is well-described by a single spectral energy distribution (SED). The relevant parameters are:

    + ``pop_sed``
    + ``pop_yield``
    + ``pop_yield_units``
    + ``pop_Emin``
    + ``pop_Emax``
    + ``pop_EminNorm``
    + ``pop_EmaxNorm``
    
See the :doc:`params` page for information about the possible values of these parameters.

``pop_sed_model=False``
~~~~~~~~~~~~~~~~~~~~~~
In this case, the SED of sources will not be considered in detail. Instead, the amount of radiation emitted in the Lyman-Werner, Lyman-continuum, and X-ray bands is determined by independent parameters.

Option 1: ``pop_fstar`` is not ``None``
In this case, the following parameters are fair game:

    + ``pop_Nion``
    + ``pop_fesc``
    + ``pop_Nlw``
    + ``pop_cX``
    + ``pop_fX``




Option 2: ``pop_fstar`` is ``None``
In this case, only three parameters are relevant:

    + ``pop_xi_LW``
    + ``pop_xi_UV``
    + ``pop_xi_XR``
    


