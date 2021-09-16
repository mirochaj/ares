:orphan:

More General Star Formation Histories and Spectral Synthesis
============================================================
By default, *ARES* does not perform spectral synthesis. For example, in :doc:`example_pop_galaxy`, the luminosity of galaxies is determined using a fixed conversion factor between :math:`1600\unicode{x212B}` luminosity and star formation rate (SFR). The proper way to do this is to sum the luminosity from stars of all ages. In practice, the constant conversion factor works well for rest UV studies, since the rest UV depends largely on massive, short-lived stars. However, for non-trivial star formation histories (SFHs), or predictions at longer wavelengths, performing spectral synthesis in details is a must.

To enable a more general treatment of galaxy growth histories, including the proper synthesis of their spectra, we must use the ``GalaxyEnsemble`` object, which supercedes the ``GalaxyCohort`` object. We describe this object in more detail below.

Setting up a ``GalaxyEnsemble`` object
--------------------------------------
We can build a new population object by modifying just a few parameters relative to the "base" double power-law SFE approach taken in `Mirocha, Furlanetto, & Sun (2017) <http://adsabs.harvard.edu/abs/2017MNRAS.464.1365M>`_:

::

	# Extract base set of parameters using a double power-law SFE
	pars = ares.util.ParameterBundle('mirocha2017:base').pars_by_pop(0, 1)

	# Add modifications necessary to handle generalizations
	pars['pop_sfr_model'] = 'ensemble'
	pars['pop_aging'] = True
	pars['pop_ssp'] = True
	
The ``pop_sfr_model`` setting ensures that the correct model is used, while ``pop_aging`` and ``pop_ssp`` force *ARES* to properly track the impact of aging on the spectra of galaxies, by treating star formation at each timestep as a "simple stellar population," i.e., a burst, which ages passively from that point onward.

For illustrative purposes, let's build two model galaxies: one with an exponentially rising (but noisy) SFH, and another that is identical for the first 900 Myr of evolution, but then is suddenly switched off:
	
::

	tarr = np.arange(50, 1001, 1.)       # array of times in Myr
	
	# Exponentially-rising SFH with log-normal scatter
	sfh1 = np.exp(tarr / 200.)
	sfh1 *= np.random.lognormal(size=tarr.size, sigma=0.5)
	sfh1 = np.atleast_2d(sfh1)

	# For contrast, compare to same SFH that is nulled for last 50 Myr
	sfh2 = sfh1.copy()
	sfh2[:,tarr > 950] = 0.0
	
	# Plot histories for sanity check
	pl.figure(1)
	pl.plot(tarr, sfh1[0], color='k')
	pl.plot(tarr, sfh2[0], color='b')
	pl.xlabel(r'$t / \mathrm{Myr}$')
	pl.ylabel(r'SFR')

.. note :: We've made the SFH arrays 2-D because in general we can perform 
	spectral synthesis on an entire population of galaxies all at once. The 
	first dimension of the SFH arrays corresponds to galaxy ID number.

Now, to pass these histories to *ARES* directly and bypass all the usual SFH-generating machinery, use the ``pop_histories`` parameter:

::

	pars1['pop_histories'] = {'t': tarr, 'sfh': sfh1, 'nh': np.ones_like(sfh1)}
	pars2['pop_histories'] = {'t': tarr, 'sfh': sfh2, 'nh': np.ones_like(sfh2)}

Note that we've added ``nh`` as well, the number density of halos. In this example, since we're treating individual galaxies, this is set to unity. However, in general, one can treat the evolution of galaxies in halo mass bins, in which case ``nh`` may be set to the number density of the DM parent halos of these galaxies.

Now, create a few ``GalaxyPopulation`` instances:

::

	pop1 = ares.populations.GalaxyPopulation(**pars1)
	pop2 = ares.populations.GalaxyPopulation(**pars2)


All the spectral synthesis machinery lives in ``ares.util.SpectralSynthesis``, which is initialized as an attribute ``synth`` belonging to each ``GalaxyEnsemble`` instance. So, to plot the spectra of our two idealized galaxies at :math:`t=1` Gyr, we can do:
	
::

	# Just look at the rest UV for now.
	waves = np.arange(800, 3000, 10)
	
	# Generate spectra in rest frame
	spec1 = pop1.synth.Spectrum(sfh=sfh1[0], waves=waves, tarr=tarr, tobs=1000)
	spec2 = pop2.synth.Spectrum(sfh=sfh2[0], waves=waves, tarr=tarr, tobs=1000)

	# Plot
	pl.semilogy(waves, spec1, color='k')
	pl.semilogy(waves, spec2, color='b')
	pl.xlabel(r'$\lambda / \AA$')
	pl.ylabel(r'$f_{\nu}$')
	
.. note :: These spectra correspond to the `BPASS <http://bpass.auckland.ac.nz/>`_ v1.0 models, since that is the 
	default in the ``mirocha2017`` parameter bundle. You can switch to *starburst99* by setting ``pop_sed='leitherer1999'``. You can also change the stellar metallicity via the ``pop_Z`` parameter. If you have a model for metal enrichment, that is possible to supply as well.
	
There are many options for outputting photometry in addition to / instead of rest spectra. Contact me if you're interested in these features as they are not yet documented.
	
Using the ``GalaxyEnsemble`` from within *ARES*
-----------------------------------------------
In practice, you may want to leverage the features of the ``GalaxyEnsemble`` object from within an *ARES* simulation, e.g., the 21-cm signal, metagalactic gackground, or while modeling a population of galaxies and comparing to observed UV luminosity functions or stellar mass functions.

Once again, contact me if you're interested in these features as they are not yet documented.

