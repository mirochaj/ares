:orphan:

Star Clusters
=============
One of the main simplifying assumptions in our main galaxy evolution model (see :doc:`example_pop_galaxy`) is that the UV luminosity of sources traces only the current star formation. This is in general a pretty good approximation because galaxy star formation histories (at least in our models) are rising rapidly with time, so the star formation on each successive time-step is greater than in those intervals preceding it. This will *not* be a good approximation for sources with more gradual star formation histories. In an extreme example, simple stellar populations (SSPs), i.e., those that form in an instantaneous burst and evolve in color and luminosity passively afterward as stars of different masses evolve, this approximation will be *really* bad. 

The ``ClusterPopulation`` is an attempt to enable a proper treatment of the aging of stellar populations. Similar features may eventually be incorporated into the ``GalaxyCohort`` model as well, but for now they remain separate.

A few usual imports before we begin:

::

    import ares
    import numpy as np
    import matplotlib.pyplot as pl


Globular Clusters as a Working Example
--------------------------------------
A simple model for a globular cluster's emission can be extracted from one's favorite stellar population synthesis model. For example, 

::

    src = ares.sources.SynthesisModel(source_sed='eldridge2009', source_ssp=True)
    
Setting ``source_ssp`` to ``True`` recovers the spectral evolution of an instantaneous burst of star formation, which is one of the main products of codes like BPASS and Starburst99. The main attributes of this class are:

    * ``times``: Array of times at which we have model spectra (in Myr)
    * ``wavelengths``: Array of wavelengths for which we have spectra (in :math:`\AA`)
    * ``data``: 2-D array containing the specific luminosity of the stellar population as a function of wavelength and time (in that order).
    
So, to plot the evolution of the UV luminosity (at :math:`1500 \AA`, let's say) of a :math:`10^5 M_{\odot}` star cluster, we could do something like

::

    i = np.argmin(np.abs(1500. - src.wavelengths))
    pl.loglog(src.times, src.data[i,:] * 1e5)
    
By default, *ARES* normalizes the spectra to be specific luminosities per solar mass of star formation, hence the factor of :math:`10^5` in the second line. 

Now, this seems all well and good, but what if we wanted to study a population of star clusters forming across the Universe with some mass distribution and perhaps some evolution in the rate at which such objects form over time. This is where the ``ClusterPopulation`` object comes into play. Essentially all it is doing is generating a whole population of such objects and integrating their spectra over time, while being careful to weight by the relative number of objects as a function of their mass (and thus overall luminosity).

The key quantities needed to model such a population include:

    * The mass distribution of objects, i.e., how many clusters do we get as a function of mass.
    * The formation rate density of objects, i.e., how many clusters form per unit volume as a function of time (or redshift).
    * The metallicity of clusters, and any changes to their IMF or presence/absence of binary populations (in BPASS, at least). 
    
Let's construct a dictionary of parameters that describes a simple population of star clusters, that we assume form at a constant rate (per unit volume) and follow a log-normal distribution in mass:

::

    pars = \
    {
     # 1 cluster / Gyr / cMpc^3 between 10 <= z <= 25. Stop aging at z=3.
     'pop_frd': lambda z: 1e-9,            
     'pop_zform': 25.,
     'pop_zdead': 10.,
     'final_redshift': 3.,
          
     # Cluster mass distribution
     'pop_mdist': 'pq[0]',
     'pq_func[0]': 'lognormal',
     'pq_func_var[0]': 'M',
     'pq_func_par0[0]': 1.0,
     'pq_func_par1[0]': np.log10(1e5), # i.e., xi=1
     'pq_func_par2[0]': 0.5,
     
     # Cluster MF resolution and range
     'pop_dlogM': 0.05,
     'pop_Mmin': 1e3,
     'pop_Mmax': 1e8,
     
     # Cluster SED
     'pop_sed': 'eldridge2009',
     'pop_Z': 1e-3,
     'pop_rad_yield': 'from_sed',
     'pop_Emin': 1.,
     'pop_Emax': 24.6,
     
     # A few switches to make sure these objects act like clusters
     'pop_aging': True,
     'pop_ssp': True,
     'pop_age_res': 10, # Myr
    }
    
A few things of note here. First, we used a constant formation rate density (``pop_frd``) but can easily generalize to a more complex function. Second, we used the :doc:`uth_pq` to create the cluster mass distribution (via ``pop_mdist``). Finally, we opted for BPASS (``pop_sed='eldridge2009'``) with low metallicity (:math:``Z=0.001``).

To go ahead and create the population, we first import the necessary class,

::
    
    from ares.populations.ClusterPopulation import ClusterPopulation

and then create an instance of it,

::
    
    cpop = ClusterPopulation(**pars)
    
Let's first verify that this population has the properties we said it should, e.g., by looking at the star formation rate density (should be a constant) and the mass function (should be log-normal):

::

    pl.figure(1)
    
    z = np.arange(5, 40)
    pl.plot(z, cpop.SFRD(z))
    pl.xlabel(r'$z$')
    pl.ylabel(ares.util.labels['sfrd'])
    
Internally, *ARES* uses *cgs* units, which is why the star-formation rate density (SFRDs) here is so small (it's in :math:`\mathrm{g} \ \mathrm{s}^{-1} \ \mathrm{cm}^3`). 

Now, for the mass function:
    
::
    
    pl.figure(2)
    
    Marr = np.logspace(3, 8)
    pl.semilogx(Marr, cpop.MassFunction(M=Marr))    
    pl.xlabel(r'$M_{\star} / M_{\odot}$')
    pl.ylabel(r'Cluster Mass Function')


Having recovered our basic inputs, let's move on to a more complex quantity: the UV luminosity function. We should notice a change in the luminosity function at different times -- and in particular, just after new clusters stop forming (at ``pop_zdead=10``):

::
    
    pl.figure(3)
    for z in [6, 8, 9, 10, 15, 20, 25]:
        mags, phi = cpop.LuminosityFunction(z=z)
        pl.semilogy(mags, phi, label=r'$z={}$'.format(z))
    
    # Tidy up a bit
    pl.ylim(1e-7, 1)
    pl.xlim(-25, 0)
    pl.legend(loc='upper left', frameon=True, fontsize=14)
    pl.xlabel(r'$M_{\mathrm{UV}}$')
    pl.ylabel(ares.util.labels['galaxy_lf'])


For a discussion of the shape of the GC luminosity function, see, e.g., `Boylan-Kolchin (2018) <http://adsabs.harvard.edu/doi/10.1093/mnras/sty1490>`_. To contrast it with the luminosity function of ''normal'' high-z galaxies for yourself, see :doc:`example_pop_galaxy`.


Using Star Clusters in *ARES* Simulations
-----------------------------------------
A good place to start here is the example on :doc:`example_gs_multipop`. In this example, one could simply replace the PopIII source population with a globular cluster population, being sure to include the X-ray emission from GCs as a separate source population. It would also be wise to upgrade the simple PopII source prescription in that example with :doc:`example_pop_galaxy`, since the use of GCs implies an interest in luminosity functions at high-:math:`z`.

