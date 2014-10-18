Initializing a Stellar Source
=============================
In these examples, we will initialize a stellar population object, defined
by the minimum virial temperature of halos in which stars form, :math:`T_{\text{min}}`, 
the star formation efficiency, :math:`f_{\ast}`, and perhaps other things.

To begin, import glorb and initialize an instance of the :class:`StellarPopulation<glorb.populations.StellarPopulation>` class:

:: 

    import glorb
    pop = glorb.populations.StellarPopulation(Tmin=1e4, fstar=0.1)
    
Once initialized, there are several class methods available to compute the star-formation rate density (SFRD) and emissivity (in the UV and X-ray):
    
::

    z = 20.
    pop.SFRD(z)                          # [g / ccm**3 / s]
    pop.XrayLuminosityDensity(z)         # [erg / ccm**3 / s]
    pop.LymanWernerLuminosityDensity(z)  # [erg / ccm**3 / s]

    