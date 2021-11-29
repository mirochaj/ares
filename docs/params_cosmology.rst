:orphan:

Cosmology Parameters
====================
The default cosmological parameters in *ARES* are from *Planck*. Specifically, we take values from the last column of Table 4 in `Planck XIII <http://adsabs.harvard.edu/abs/2015arXiv150201589P>`_. 

.. note :: Several input files (e.g., lookup tables for the halo mass           
    function, initial conditions, etc.) depend on these vales. There currently 
    is not a system in place to make sure there is a match between the      
    parameters you pass at run-time and the lookup tables read-in from disk. 
    Beware!

``omega_m_0``
    Matter density, relative to the critical density.
    
    Default: :math:`\Omega_{m,0} = 0.3089`

``omega_b_0``
    Baryon density, relative to the critical density.

    Default: :math:`\Omega_{b,0} = 0.0486`

``omega_l_0``
    Dark energy density, relative to the critical density.
    
    Default :math:`\Omega_{\Lambda,0} = 0.6911`
    
``hubble_0``
    Hubble parameter today.
    
    Default: 0.6774 :math:`[100 \ \text{km} \ \text{s}^{-1} \ \text{Mpc}^{-1}]`

``helium_by_mass``
    Fractional helium abundance by mass.
    
    Default: 0.2453

``cmb_temp_0``
    Temperature of the cosmic microwave blackbody, today.
    
    Default: 2.7255 :math:`[\text{K}]`
    
``sigma_8``    
    Default: 0.8159

``primordial_index``
    Default: 0.9667
    
    