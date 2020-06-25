:orphan:

Self-Consistent Dust Reddening
==============================
In `Mirocha, Mason, & Stark (2020) <https://ui.adsabs.harvard.edu/abs/2020arXiv200507208M/abstract>`_, we describe a simple extension to the now-standard galaxy model in ARES that generates ultraviolet colours self-consistently, rather than invoking IRX-:math:`\beta` relations to perform a dust correction. Here, we describe how to work with these new models.

Preliminaries
~~~~~~~~~~~~~
Before getting started, two lookup tables are required that don't ship by default with ARES (via the ``remote.py`` script; see :doc:`install`):

- A new halo mass function lookup table that employs the Tinker et al. 2010 results, rather than Sheth-Tormen (used in most earlier work with ARES).
- A lookup table of halo mass assembly histories.

To create these yourself, you'll need the `hmf <https://github.com/steven-murray/hmf>`_ code, which is installable via pip.

This should only take a few minutes even in serial. First, navigate to ``$ARES/input/hmf``, where you should see a few ``.hdf5`` files and Python scripts. Open the file ``generate_hmf_tables.py`` and make the following adjustments to the parameter dictionary:

::

	"hmf_model": 'Tinker10',
	
	"hmf_tmin": 30,
	"hmf_tmax": 2e3,
	"hmf_dt": 1,	
	
	"cosmology_id": 'paul',
	"cosmology_name": 'user',
	"sigma_8": 0.8159, 
	'primordial_index': 0.9652, 
	'omega_m_0': 0.315579, 
	'omega_b_0': 0.0491, 
	'hubble_0': 0.6726,
	'omega_l_0': 1. - 0.315579, 
	
	
The new HMF table will use constant 1 Myr time-steps, rather than the default redshift steps, and employ a cosmology designed to remain consistent with another project (led by a collaborator whose name you can probably guess)!

Once you've run the ``generate_hmf_tables.py`` script, you should have a new file, ``hmf_Tinker10_user_paul_logM_1400_4-18_t_1971_30-2000.hdf5``, sitting inside ``$ARES/input/hmf``. Now, we're almost done. Simply execute:

::

	python generate_halo_histories.py hmf_Tinker10_user_paul_logM_1400_4-18_t_1971_30-2000.hdf5
	
The additional resulting file, ``hgh_Tinker10_user_paul_logM_1400_4-18_t_1971_30-2000_xM_10_0.10.hdf5``, will be found automatically in subsequent calculations.


Example
~~~~~~~
With the required lookup tables now in hand, we can start in the usual way:

::

	import ares
	import numpy as np
	import matplotlib.pyplot as pl

and read-in the best-fit parameters via

::

	pars = ares.util.ParameterBundle('mirocha2020:univ')
	
	
Now, we create a ``GalaxyPopulation`` object,

::

	pop = ares.populations.GalaxyPopulation(**pars)
	

which is an instance of the ``GalaxyEnsemble`` class, rather than the ``GalaxyCohort`` class, the latter of which is described in :doc:`example_galaxy_pop`. This offers a more general approach, including more complex star formation histories and proper treatment of spectral synthesis.

We can plot the UVLF in the usual way,

::

    # First, some data
    obslf = ares.analysis.GalaxyPopulation()
    ax = obslf.PlotLF(z=6, round_z=0.2)
    
    # Now, the predicted/calibrated UVLF
    mags = np.arange(-25, -5, 0.1)
    phi = pop.LuminosityFunction(6, mags)
    
    ax.semilogy(mags, phi)
	
The main difference between these models and the simpler models (from, e.g., the ``mirocha2017`` parameter bundle) is access to UV colours. The following high-level routine will make a plot like Figure 4 from the paper:

::

	axes = obslf.PlotColors(pop, fig=2)
	
This will take order :math:`\simeq 10` seconds, as modeling UV colours requires synthesizing a reasonbly high resolution (:math:`\Delta \lambda = 20 \unicode{x212B}` by default) spectrum for each galaxy in the model, so that there are multiple points within photometric windows. You can adjust the keyword arguments ``z_uvlf`` and ``z_beta`` to see different redshifts, while ``sources`` will control the datasets being plotted.

.. note :: Computing colors from fits in the `Calzetti et al. 1994 <https://ui.adsabs.harvard.edu/abs/1994ApJ...429..582C/abstract>`_ windows requires higher resolution, given that these windows are each only :math:`\Delta \lambda \sim 10 \unicode{x212B}` wide. Adjust the ``dlam`` keyword argument to increase the wavelength-sampling used prior to photometric UV slope estimation.

To access the magnitudes and colours more directly, you can do something like

::

    mags = pop.Magnitude(6., presets='hst')
    beta = pop.Beta(6., presets='hst')
    
    fig3, ax3 = pl.subplots(1, 1, num=3)
    ax3.scatter(mags, beta, alpha=0.1, color='b', edgecolors='none')

which computes the UV slope :math:`\beta` using the *Hubble* filters appropriate for the input redshift (see Table A1 in the paper).

.. note :: Other ``presets`` include ``'jwst'``, ``'jwst-w'``, ``'jwst-m'``, and ``'calzetti1994'``.

You can bin points in the :math:`M_{\text{UV}}-\beta` plane, if you prefer it, via the ``return_binned`` keyword argument,

::

	mags = np.arange(-25, -10, 0.5) # bin centers
	beta, beta_s = pop.Beta(6., presets='hst', Mbins=mags, return_binned=True,
		return_scatter=True)
	
	# Plot scatter in each MUV bin as errorbars
	ax3.errorbar(mags, beta, yerr=beta_s.T, color='b', marker='s', fmt='o')

Recall that each galaxy in the model actually represents an "average" galaxy in some halo mass bin, i.e., there is not a 1:1 correspondence between galaxies and elements in ``mags`` and ``beta`` above, which is why we generally weight by halo abundance in each bin. The default mass-bins have :math:`\Delta \log_{10} M_h = 0.01` wide, within which ARES inject ``pop_thin_hist`` halos and down-weights their abundance accordingly.

Lastly, to extract the ``raw'' galaxy population properties, like SFR, stellar mass, etc., you can use the ``get_field`` method, e.g.,

::

	Ms = pop.get_field(6., 'Ms') # stellar mass
	Md = pop.get_field(6., 'Md') # dust mass
	Sd = pop.get_field(6., 'Sd') # dust surface density
	# etc.
	
To see what's available, check out

::

	pop.histories.keys()
	

From these simple commands, most plots and analyses from the paper can be reproduced in relatively short order.
	
	
Notes on performance
~~~~~~~~~~~~~~~~~~~~
The compute time for these models is determined largely by three factors:

* The number of halos (really halo bins) we evolve forward in time. The default ``mirocha2020:univ`` models use :math:`\Delta \log_{10} M_h = 0.01` mass-bins, each with 10 halos (``pop_thin_hist=10``). For a larger sample, e.g., when lots of scatter is being employed, larger values of ``pop_thin_hist`` may be warranted.
* The number of observed redshifts at which :math:`\beta` is synthesize, since new spectra must be generated.
* The number of wavelengths used to sample the intrinsic UV spectrum of galaxies. When computing :math:`\beta` via photometry (Hubble or JWST), :math:`\Delta \lambda = 20 \unicode{x212B}` will generally suffice. However, 

For the models in `Mirocha, Mason, & Stark (2020) <https://ui.adsabs.harvard.edu/abs/2020arXiv200507208M/abstract>`_, with :math:`\Delta \log_{10} M_h = 0.01 (``hmf_logM=0.01``), ``pop_thin_hist=10``, calibrating at two redshifts for :math:`\beta` (4 and 6), with :math:`\Delta \lambda = 20 \unicode{x212B}`, each model takes :math:`\sim 10-20` seconds, depending on your machine.

.. note :: Generating the UVLF is essentially free compared to computing :math:`\beta`.

For more information on what's happening under the hood, e.g., with regards to spectral synthesis and noisy star-formation histories, see :doc:`example_pop_sps`.



