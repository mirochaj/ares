Using Data from the Literature
==============================
Within ``$ARES/input/litdata`` there are several empirical formulae and datasets
gathered from the literature, which typically include fits to the cosmic
star-formation rate density with redshift, the galaxy or quasar luminosity
function, and/or model spectral energy distributions.

The current list of papers currently included (at least to some extent) are:

* `Leitherer et al. (1999) <http://adsabs.harvard.edu/abs/1999ApJS..123....3L>`_ (``'leitherer1999'``)
* `Ueda et al. (2003) <http://adsabs.harvard.edu/abs/2003ApJ...598..886U>`_ (``'ueda2003'``)
* `Sazonov et al. (2004) <http://adsabs.harvard.edu/abs/2004MNRAS.347..144S>`_ (``'sazonov2004'``)
* `Haardt & Madau (2012) <http://adsabs.harvard.edu/abs/2012ApJ...746..125H>`_  (``'haardt2012'``)
* `Ueda et al. (2014) <http://adsabs.harvard.edu/abs/2014ApJ...786..104U>`_ (``'ueda2014'``)
* `Robertson et al. (2015) <http://adsabs.harvard.edu/abs/2015ApJ...802L..19R>`_  (``'robertson2015'``)
* `Aird et al. (2015) <http://arxiv.org/abs/1503.01120>`_ (``'aird2015'``)

Notice that the shorthand for these papers are just the first author's last 
name and the year of publication.

For the rest of the examples on this page, we'll assume you've already imported *ares*, i.e. you've executed:

::  

    import ares

To read the data from e.g., Haardt & Madau (2012), simply do:

::
    
    hm12 = ares.util.read_lit('haardt2012')

Then, access functions for e.g., the SFRD via

::

    hm12.SFRD(6)  # in Msun / yr / cMpc**3

or the quasar luminosity function:

::

    u03 = ares.util.read_lit('ueda2003')
    u03.LuminosityFunction(1e42, z=0)
    
See ``$ARES/tests/lit`` for more examples.

Using models from the literature in RT calculations
---------------------------------------------------
The motivation for developing a generalized interface to data from the literature is to (1) enable development of a common set of analysis routines to facilitate
apples-to-apples comparisons between various results and the literature, and
(2) to ease the integration of these models with radiative transfer calculations.

For example, to initialize an *ares* object for simple inspection, you could do something like: 

::
        
    pop = ares.populations.GalaxyPopulation(source_type='bh', approx_xrb=False, 
        source_sed='sazonov2004', pop_lf='ueda2003')
    
If the value of ``pop_lf`` is not ``'schecter'`` or a Python function, it will assume you are trying to use a model from the literature. 

.. note :: These parameter names are subject to change / deprecation.        
    
Expanding the Database
----------------------
If you'd like to add your favorite empirical formulae from the literature to *ares*, here are a few conventions to follow:

File contents:

- Dictionaries containing best-fit parameter values and :math:`1-\sigma` error bars.
- Functions for computing the SFRD, LuminosityFunction, Emissivity, and/or Spectrum.

Naming conventions:

- Your file should be called `<last name of first author><year of publication>.py`.
- As long as this file lives in ``$ARES/input/litdata``, *ares* will find it.
- Keep best-fit parameter values and errors stored in dictionaries.




