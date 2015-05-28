Galaxy Populations
======================
In these examples, we will initialize a galaxy population object, defined
by... and other optional keyword arguments.

To begin, import ares and initialize an instance of the GalaxyPopulation class:

:: 

    import ares
    pop = ares.populations.GalaxyPopulation()
    
Once initialized, there are several class methods available to compute the star-formation rate density (SFRD) and emissivity (in the UV and X-ray):
    
::

    z = 20.
    pop.LuminosityFunction(z)            # [g / ccm**3 / s]
    pop.SpaceDensity(z)                  # [erg / ccm**3 / s]
    

Under construction!

