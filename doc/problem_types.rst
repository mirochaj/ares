:orphan:

Problem Types
=============
There are several pre-defined problem types one can access via the parameter
``ptype``. Note that you can also grab the parameters for a given problem type using the :doc:`param_bundles` machinery. For example,

::

    import ares
    pars = ares.util.ParameterBundle('prob:101')
    
will return a dictionary of parameters for ``problem_type=101``.


``ptype`` :math:`\leq 20`
--------------------------
These are all 1-D radiative transfer problems. Will document eventually!

            
``ptype`` :math:`\geq 100`
--------------------------
These are all uniform background / reionization / global 21-cm problems.

100
~~~
Blank slate global 21-cm signal problem -- no default populations will be initialized, and all "control parameters" take on their default values. Basically this means that the simplest solvers / assumptions will be adopted for everything. Only use this if you know what you're doing!

101
~~~
Simple global 21-cm signal problem in which the Ly-:math:`\alpha`, LyC, and X-ray production is proportional to the rate of collapse onto all halos exceeding a minimum virial temperature threshold (``pop_Tmin``) or mass (``pop_Mmin``). The main free parameters are:

    + ``pop_yield{0}``: Number of LW photons emitted per baryon of star formation. Stellar spectrum assumed flat.
    + ``pop_yield{1}``: Normalization of the X-ray luminosity star-formation rate relation in the 0.5-8 keV band.
    + ``pop_yield{2}``: Number of LyC photons emitted per baryon of star formation.
    + ``pop_fesc{2}``: Escape fraction of LyC radiation.
    + ``pop_Tmin{0}``: Minimum virial temperature of star-forming halos. Note that ``pop_Tmin{1}`` and ``pop_Tmin{2}`` are automatically linked to ``pop_Tmin{0}``.

    .. note :: In earlier versions of *ARES* these parameters were denoted more simply as ``Nlw``, ``fX``, ``Nion``, ``fesc``, and ``Tmin``. You can still use this approach (i.e., this shouldn't break backward compatibility), though in the future this may not be true. 
    
102
~~~
Slightly more advanced global 21-cm signal problem in which the Ly-:math:`\alpha`, LyC, and X-ray production is still proportional to the rate of collapse onto all halos exceeding a minimum virial temperature threshold (``pop_Tmin``) or mass (``pop_Mmin``), but the photon production efficiencies are calculated from a stellar synthesis model. The main difference between this problem and problem 101 is that the LW and LyC efficiencies are no longer independent. As a result, there are only *two* source populations: one stellar and one for X-rays. The main parameters are slightly different as a result:

    + ``pop_sed{0}``: Spectral energy distribution of stellar populations. By default, this is ``eldridge2009``, i.e., the *BPASS* version 1.0 models.
    + ``pop_Z{0}``: Stellar metallicity.
    + ``pop_fesc{0}``: Escape fraction of LyC radiation.
    + ``pop_yield{1}``: Normalization of the X-ray luminosity star-formation rate relation in the 0.5-8 keV band.
    + ``pop_Tmin{0}``: Minimum virial temperature of star-forming halos. Note that ``pop_Tmin{1}`` is automatically linked to ``pop_Tmin{0}``.


    




    