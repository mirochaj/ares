Reading Data from the Literature
--------------------------------
Within ``$ARES/input`` there are several empirical formulae and datasets
gathered from the literature, which typically include fits to the cosmic
star-formation rate density with redshift, the galaxy or quasar luminosity
function, or model spectral energy distributions.

The current list of papers currently included are:

* `Robertson et al. (2015) <http://adsabs.harvard.edu/abs/2015ApJ...802L..19R>`_, (`robertson2015`)
* `Haardt & Madau (2012) <http://adsabs.harvard.edu/abs/2012ApJ...746..125H>`_, (`haardt2012`)

Notice that the shorthand for these papers are just the first author's last 
name and the year of publication.

To access the data, simply do:

::

    import ares
    
    hm12 = ares.util.read_lit('haardt2012')
    
Then, access functions for e.g., the SFRD via

::

    hm12.sfrd(6)  # in Msun / yr / cMpc**3
    
    
Adding More
-----------
Under construction...

