Initializing a Black Hole Source
================================
In these examples, we will initialize a black hole population object, defined
by the minimum virial temperature of halos in which stars form, Tmin, the star formation efficiency, fstar, and other optional keyword arguments.

To begin, import glorb and initialize an instance of the BlackHolePopulation class:

:: 

    import glorb
    pop = glorb.populations.BlackHolePopulation(Tmin=1e4, fstar=0.1)
    
Once initialized, there are several class methods available to compute the star-formation rate density (SFRD) and emissivity (in the UV and X-ray):
    
::

    z = 20.
    pop.AccretionRateDensity(z)          # [g / ccm**3 / s]
    pop.XrayLuminosityDensity(z)         # [erg / ccm**3 / s]
    pop.LymanWernerLuminosityDensity(z)  # [erg / ccm**3 / s]
    
Class methods always return values in cgs units, and when applicable, volume densities are assumed to be in comoving units (in the comments above, ccm stands for co-moving centimeters).

To convert to more recognizable units, use conversion factors from rt1d:

::

    from rt1d.physics.Constants import rhodot_cgs
    pop.SFRD(z) * rhodot_cgs                              # [Msun / cMpc**3 / yr]
    pop.XrayLuminosityDensity(z) * cm_per_mpc**3          # [erg / cMpc**3 / s]
    
where Msun is solar masses, and cMpc is used to denote co-moving Megaparsecs.


Black hole models have a wider variety of behaviors available than stellar models.

