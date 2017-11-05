:orphan:

Including Population III Stars
==============================
One of the generic results of using :doc:`example_galaxypop` is that they tend to produce strong late-time absorption troughs in the global 21-cm signal (this was the point of `Mirocha, Furlanetto, & Sun (2017) <http://adsabs.harvard.edu/abs/2017MNRAS.464.1365M>`_). Our interpretation was that deviations from these predictions could indicate "new" source populations, like Population III stars and their remnants. Indeed, we found some evidence that such objects introduce asymmetries in the global 21-cm signal (see `Mirocha et al., in review, <https://arxiv.org/abs/1710.02530>`_ for details). 

The PopIII stars in this paper are simple, and were designed to seamlessly integrate with *ares* while capturing the general behavior of more detailed models (e.g., `Mebane, Mirocha, \& Furlanetto, submitted <https://arxiv.org/abs/1710.02528>`_). This section will describe how to make use of these models yourself.

The easiest way to tap into these models is via the `ParameterBundle` framework. To begin,

::

    import ares

then, for example,

::

    pars = ares.util.ParameterBundle('mirocha2016:dpl') \
         + ares.util.ParameterBundle('mirocha2017:high')
         
Parameter bundles are designed to be added together, so as to build-up more complex calculations piece by piece. The above snippet takes the default model from `Mirocha, Furlanetto, & Sun (2017) <http://adsabs.harvard.edu/abs/2017MNRAS.464.1365M>`_ and adds on the default PopIII model from `Mirocha et al., in review, <https://arxiv.org/abs/1710.02530>`_. The "high" suffix refers to the mass of the PopIII stars -- in this case, high means :math:`\sim 100 \ M_{\odot}`. There are also bundles for "low" and "med" mass PopIII stars, which just changes the mass (and resultant spectra) according to the `Schaerer (2002) <http://adsabs.harvard.edu/abs/2002A%26A...382...28S>`_ models. 

You'll notice that while the 'mirocha2016:dpl' bundle contains parameters for two source populations -- one that provides the UV emission and another that produces X-rays -- the addition of the PopIII bundle adds two more populations, again one each for UV and X-ray. You can customize the properties of these sources further via the following parameters:

* `pop_sfr{2}` 
    The typical star formation rate (SFR) in PopIII halos (in :math:`M_{\odot} \ \mathrm{yr}^{-1}`), in a mass-bin-averaged sense (i.e., we assume PopIII star formation occurs in discrete bursts so the SFR in any individual halo is ill-defined).
* `pop_time_limit{2}`
    The typical length of the PopIII phase (in Myr).
* `pop_bind_limit{2}` 
    The typical binding energy (in erg) at which point halos transition to PopII star formation .
* `pop_rad_yield{3}`
    The X-ray production efficiency in PopIII halos (in :math:`\mathrm{erg} \ \mathrm{s}^{-1} \ (M_{\odot} \ \mathrm{yr}^{-1})^{-1}). 
    
It is possible to use a halo mass-dependent prescription for the PopIII SFR if you'd like. In that case, you'll need to update ``pop_sfr_model{2}`` to be ``sfe-func``. See :doc:`example_galaxypop` for a reminder on how to do that.

Note on Feedback
~~~~~~~~~~~~~~~~
One of the defining features of PopIII sources is their susceptibility to the global Lyman-Werner (LW) background, which drives up the minimum halo mass for star formation and thus limits the PopIII star formation rate density (SFRD). Because the LW background depends on the SFRD, the introduction of PopIII sources means *ares* calculations must be performed iteratively. As a result, **you will notice that these models can be quite a bit slower than normal *ares* calculations (by a factor of a few up to an order of magnitude, typically.**

There is some control, here, however. If you're not looking for much accuracy, you can change the default set of convergence criteria to accelerate things.


* ``feedback_LW_mean_err``
    If ``True``, calculations will terminate as soon as the mean error in :math:`M_{\min}` or the PopIII SFRD satisfy the set tolerances. By default, it is ``False``.
* ``feedback_LW_sfrd_rtol`` 
    The relative tolerance needed in order for calculations to terminate. By default, this is ``0.05``.
* ``feedback_LW_sfrd_atol`` 
    The absolute tolerance needed in order for calculations to terminate. By default, this is ``0.0`` (i.e., unused).
    
* ``feedback_LW_maxiter``
    Maximum number of iterations allowed. By default, 50.

I do have some tricks for speeding things up more, for example by using a look-up table of pre-computed initial guesses for the SFRD that results for particular parameter combinations. If this is something you're interested in, do email me.
