:orphan:

Halo Mass Function Parameters
=============================
The halo mass function is at the core of many calculations related to the high-z universe. glorb uses `hmf <https://github.com/steven-murray/hmf>`_, a halo mass function calculator written by Stephen Murray.

``hmf_func``
    Which fit to the halo mass function should be used?
    
    Options:
    
    + ``'PS'``: `Press & Schecter (1974) <http://adsabs.harvard.edu/abs/1974ApJ...187..425P>`_
    + ``'ST'``: `Sheth & Tormen (1999) <http://adsabs.harvard.edu/abs/1999MNRAS.308..119S>`_

    Default: ``'PS'``
    
    .. note :: You can actually supply any of the options allowed by the *hmf*            
        code here (for the parameter ``mf_fit``). Just be aware that not every 
        fit to the halo mass function in the literature is meant to work at 
        high redshifts!

``hmf_table``
    Path to a halo mass function lookup table.
    
    Default: ``None``
    
``hmf_analytic``
    Compute collapsed fraction, :math:`f_{\text{coll}}`, analytically? Only possible if ``fitting_function='PS'``. Useful for testing numerical integration of the mass function.
    
    Default: ``False``
    
``hmf_load``
    Search ``$ARES/input/hmf`` for halo mass function lookup table?
    
    Default: ``True``
    
``hmf_logMmin``
    Base-10 logarithm of the minimum halo mass to consider.
    
    Default: 4

``hmf_logMmax``
    Base-10 logarithm of the maximum halo mass to consider.

    Default: 16  

``hmf_dlogM``
    Base-10 logarithm of the mass resolution in halo mass function lookup table.
    
    Default: 0.05
    
``hmf_zmin``
    Minimum redshift in lookup table.

    Default: 4

``hmf_zmax``
    Maximum redshift in lookup table.
    
    Default: 80
    
``hmf_dz``
    Redshift resolution in lookup table.
    
    Default: 0.05
    