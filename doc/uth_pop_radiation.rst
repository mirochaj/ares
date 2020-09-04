:orphan:

Models for Radiation Emitted by Galaxies
========================================
There are three main ways to model the radiation emitted by galaxies, governed largely by whether or not ``pop_sed_model`` is ``True`` or  ``False``. 

``pop_sed_model=True``
~~~~~~~~~~~~~~~~~~~~~~~~~
In this case, we're assuming that the source population is well-described by a single spectral energy distribution (SED). The relevant parameters are:

    + ``pop_sed``
    + ``pop_rad_yield``
    + ``pop_rad_yield_units``
    + ``pop_Emin``
    + ``pop_Emax``
    + ``pop_EminNorm``
    + ``pop_EmaxNorm``
    
We'll return to these parameters in more detail below.

``pop_sed_model=False``
~~~~~~~~~~~~~~~~~~~~~~~~~
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
    

Available Models for Source Spectral Energy Distributions
---------------------------------------------------------
If ``pop_sed_model=True``, we of course have some decisions to make, e.g.:

- What is the appropriate spectral energy distribution (SED) for the source population I'm interested in?
- How should I normalize that SED, i.e., how much energy do sources of this type produce, and in what band?

Let's run through some common choices. For simplicity we'll work directly with the source spectra, which means we won't make any assumptions about star formation or anything of that sort. The way *ARES* is structured, this means we'll access objects in ``ares.sources`` directly. For more sophisticated calculations, all the source populations (``ares.populations``) are doing is initializing source objects for themselves. More on that in a bit.

.. note :: When working with source classes directly, just change the ``pop_`` prefix to ``source_``, and you'll be good to go. This will be our approach in the examples below. When you initialize population objects defined by a series of ``pop_`` parameters, *ARES* will automatically swap out the prefix when each population object initializes its source object.

Before we get going, as per usual:

::

    import ares
    import numpy as np
    import matplotlib.pyplot as pl

Power-Law Sources
~~~~~~~~~~~~~~~~~

Let's initialize a power-law source, which is about the simplest thing we can do:

::
    
    # Should switch to 'ares.sources.Generic'
    src = ares.sources.BlackHole(source_sed='pl', source_Emin=2e2, source_Emax=3e4)
    
    E = np.logspace(2, 4)
    
    pl.loglog(E, src.Spectrum(E))


By default, this is just (shocking news) a power-law. It will be automatically normalized such that the flux in the ``(source_EminNorm, source_EmaxNorm)`` band integrates to unity. By default, its slope is -1.5, but we can change that via ``source_alpha``. We can also add neutral attenuation, meant to describe a typical column density of hydrogen gas that "hardens" the intrinsic spectrum:

::

    src2 = src = ares.sources.BlackHole(source_sed='pl', source_Emin=2e2, source_Emax=3e4, source_logN=20.)

    pl.loglog(E, src2.Spectrum(E), ls='--')
    
    pl.savefig('ares_sed_pl.png')

Note that the spectrum is normalized such that its *intrinsic* emission integrates to unity in the specified normalization band. If you'd like to force the hardened spectrum to be used to set the normalization, set ``source_hardening='extrinsic'``.

.. figure::  https://www.dropbox.com/s/39zaskjqp369tqb/ares_sed_pl.png?raw=1
   :align:   center
   :width:   600

   Example power-law spectra, with (dashed) and without (solid) neutral absorption intrinsic to the source.



Black Hole Accretion Disk Spectra
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The simplest analytic model an accretion disk spectrum is the so-called multi-color disk (MCD) spectrum (`Mitsuda et al. 1984 <http://adsabs.harvard.edu/abs/1984PASJ...36..741M>`_), which gives rise to a modified black body spectrum since each annulus in the accretion disk has a different temperature. To access this spectrum in *ARES*, you can do, e.g.,

::

    pars = \
    {
     'source_mass': 10.,  
     'source_rmax': 1e3,
     'source_sed': 'mcd',
     'source_Emin': 10.,
     'source_Emax': 1e4,
     'source_logN': 18.,
    }
    
    src = ares.sources.BlackHole(**pars)
    
    pl.figure(2)
    pl.loglog(E, src.Spectrum(E), ls='-')

Real BH accretion disks often have a harder power-law tail to their emission, likely due to up-scattering of disk photons by a hot electron corona. The SIMPL model (`Steiner et al. 2009 <http://adsabs.harvard.edu/abs/2009PASP..121.1279S>`_) provides one method of treating this effect, and is included in *ARES*. It depends on the additional parameter ``source_fsc``, which governs what fraction of disk photons are up-scatter to a high energy tail (with spectral index ``source_alpha``). For example,

::

    pars['source_sed'] = 'simpl'
    pars['source_fsc'] = 0.1
    pars['source_alpha'] = -1.5

    src = ares.sources.BlackHole(**pars)

    pl.loglog(E, src.Spectrum(E), ls='-')
    
    pl.savefig('ares_sed_mcd_simpl.png')

You should see that there is a high energy tail, but also that the soft part of the spectrum has also been reduced (it is those photons that are up-scattered into the high energy tail).

You'll notice that this spectrum is a bit more computationally expensive to generate than the rest, that are effectively instantaneous. You can degrade the native resolution over which the SIMPL model is generated via the parameter ``source_dlogE`` to make things faster, but of course this will cause numerical artifacts in the spectrum. If you'd prefer to build-up a database of these spectra so that they need not be re-generated at the outset of each new calculation, navigate to ``$ARES/input/bhseds``, where you'll find a script for generating SIMPL SEDs over a crudely sampled parameter space (in ``source_fsc`` and ``source_alpha``).

Once you've got a spectrum tabulated, you can load it for a calculation via:

::

    # For example
    np.savetxt('your_2column_sed_model.txt', np.array([E, (E / 1e3)**-1.5]).T)
    x, y = np.loadtxt('your_2column_sed_model.txt', unpack=True)
    
    pars['source_sed'] = (x, y)
    src = ares.sources.BlackHole(**pars)
    
    pl.loglog(E, src.Spectrum(E), ls='--')
    
.. figure::  https://www.dropbox.com/s/aaw8qtbfpvqruxj/ares_sed_mcd_simpl.png?raw=1
   :align:   center
   :width:   600

   Comparison of MCD and SIMPL models.
    
Thanks to Greg Salvesen for contributing his Python implementation of this spectrum!

AGN Template
~~~~~~~~~~~~
Ideally, one could build a physical model over a broad range of photon energies for accreting BHs, but such functionality does not currently exist in *ARES*. However, in the meantime, you can access a template AGN spectrum presented in `Sazonov, Ostriker, \& Sunyaev 2004 <http://adsabs.harvard.edu/abs/2004MNRAS.347..144S>`_:

::

    pars = \
    {
     'source_sed': 'sazonov2004',
     'source_Emin': 0.1,
     'source_Emax': 1e6,
    }
    
    src = ares.sources.BlackHole(**pars)
    
    # This model spans a very broad range in energy
    E = np.logspace(-1, 5.5)
    
    pl.figure(3)
    pl.loglog(E, src.Spectrum(E))
    pl.savefig('ares_sed_sos04.png')
    
.. figure::  https://www.dropbox.com/s/pgdj6o75ylk4ua7/ares_sed_sos04.png?raw=1
   :align:   center
   :width:   600

   AGN template spectrum from Sazonov et al. (2004).
        
        
There is still a peak in the hard UV / X-ray, like we saw for the stellar mass BH spectra above, though it peaks at softer energies. There is also an additional peak visible at higher energies (the "Compton hump").

Stellar Population Synthesis Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You can also use *ARES* to access two popular stellar population synthesis models, `starburst99 <http://www.stsci.edu/science/starburst99/docs/default.htm>`_ (`Leitherer et al. 1999 <http://adsabs.harvard.edu/abs/1999ApJS..123....3L>`_) and `BPASS <http://bpass.auckland.ac.nz/>`_ (`Eldridge \& Stanway 2009 <http://adsabs.harvard.edu/abs/2009MNRAS.400.1019E>`_). The requisite lookup tables for each will be downloaded when you install *ARES* and run the ``remote.py`` script (see :doc:`install` for more details).

.. note :: Currently, *ARES* will only download the BPASS version 1.0 models, though there are newer version available from the BPASS website.

Right now, these sources are implemented as "litdata" modules, i.e., in the same fashion as we store data and models from the literature (see :doc:`example_litdata` for more info). So, to use them, you must set ``pop_sed`` or ``source_sed`` to ``"eldridge2009"`` and ``"leitherer1999"`` for BPASS and starburst99, respectively.

.. note :: The spectral resolution of these SED models is needlessly high for certain applications. To degrade BPASS spectra and get a slight boost in performance, you can run the script ``$ARES/input/bpass_v1/degrade_bpass_seds.py`` with a command-line argument indicating the desired spectral resolution in :math:`\unicode{x212B}`. Just be sure to also set ``pop_sed_degrade`` to this same number in subsequent calculations in order to read-in the new tables.

If you'd like to access the SPS data directly, you can do so via, e.g.,

::

	src = ares.sources.SynthesisModel(source_sed='eldridge2009', source_Z=0.02)

which will initialize a BPASS version 1.0 model with solar metallicity, :math:`Z=0.02=Z_{\odot}`. The raw data is stored in an aptly-named attribute, ``src.data``, which is a 2-D array: the first dimension corresponds to the wavelengths at which we have spectra (in Angstroms), while the second dimension is the times at which we have spectra (in Myr). To see the corresponding ``wavelengths``, and ``times``, see attributes of the same name.

So, for example, to plot the SED at a few times, you could do something like

::

    pl.figure(1)
    for i in range(3):
        t = src.times[i]
        pl.loglog(src.wavelengths, src.data[:,i], label=r'$t = {}$ Myr'.format(t))
		
    pl.legend()

or alternatively, the luminosity at a single wavelength vs. time:

::

    pl.figure(2)
    for i in range(0, 1000, 200):
        wave = src.wavelengths[i]
        pl.loglog(src.times, src.data[i,:], label=r'$\lambda = {} \AA$'.format(wave))
    pl.legend()	

	
By default, it is assumed that stars form continuously, so you should see a quick ramp-up of the luminosity in the above examples before reaching a plateau at late times. To instead focus on an instantaneous burst of star formation, we need to instead use a "simple stellar population," which we can do by setting ``source_ssp=True`` when initializing the ``SynthesisModel`` instance above.

If you already have some kind of ``ares.populations`` class instance in hand, you can access the associated SPS model via the ``src`` attribute, e.g.,

::

	src = pop.src
	
Just know that to vary the ``SynthesisModel`` parameters through ``ares.populations`` objects, you should change the parameter prefixes from ``source_`` to ``pop_``. For example, 

::

	pars = ares.util.ParameterBundle('mirocha2017:base').pars_by_pop(0, 1)
	
	pars['pop_sed] = 'eldridge2009'
	pars['pop_Z] = 0.02
	
	pop = ares.populations.GalaxyPopulation(**pars)
	src = pop.src # will be the same as in previous example

	


Normalizing the Emission of Source Populations
----------------------------------------------
In the previous section, all spectra were normalized such that the integral in the ``(source_EminNorm, source_EmaxNorm)`` band was unity. Importantly, all spectra internal to *ARES* are defined such that the function ``Spectrum`` yields a quantity proportional to the *amount of energy emitted* at the corresponding photon energy, not the number of photons emitted. 

Ultimately, we generally want to use these spectral models to create entire populations of objects, assumed to exist throughout the Universe. This is the distinction between Population objects and Source objects -- the latter know nothing about the global properties of the sources, like their star formation rate density or radiative yield (i.e., photons or energy per unit SFR).

In global 21-cm models we typically invoke a population of X-ray binaries (that live in star-forming galaxies). A simple example of such a population is explored in :doc:`example_crb_xr`.





