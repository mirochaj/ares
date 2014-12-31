Population Parameters
=====================
``formation_epoch``
    Redshift interval over which sources are "on."

    Default: (50, 0)
    
``is_lya_src`` 
    Sources contribute to Ly-:math:`\alpha` background?
    
    Default: ``True``

``is_ion_src_cgm`` 
    Sources contribute growth of HII regions?

    Default: ``True``

``is_ion_src_igm`` 
    Sources contribute ionization in bulk IGM?

    Default: ``True``
    
``is_heat_src_igm``
    Sources emit X-rays and heat bulk IGM?
    
    Default: ``True``
    
    
Star formation history
----------------------    
    
``Tmin``
    Minimum virial temperature of star-forming halos.
    
    Default: :math:`10^4` [Kelvin]
    
``Mmin``
    Minimum mass of star-forming halos. Will override ``Tmin`` if set to 
    something other than ``None``.

    Default: ``None`` [:math:`M_{\odot}`]

``fstar``
    Star formation efficiency, :math:`f_{\ast}`, i.e., fraction of collapsing
    gas that turns into stars.
    
    Default: 0.1

``sfrd``
    The star formation rate density (SFRD) as a function of redshift. If provided, will override ``Tmin`` and ``Mmin``. For example, a constant (co-moving) SFRD of :math:`1 \ M_{\odot} \ \text{yr}^{-1} \ \text{cMpc}^{-3}` would be ``sfrd=lambda z: 1.0``.
    
    Default: ``None`` [:math:`M_{\odot} \ \text{yr}^{-1} \ \text{cMpc}^{-3}`]
        
X-ray background
----------------
``cX``
    Normalization of the X-ray luminosity to star formation rate (:math:`L_X`-SFR) relation in 
    band given by ``spectrum_EminNorm`` and ``spectrum_EmaxNorm``. If ``approx_xrb=1``, this
    represents the X-ray luminosity density per unit star formation, such that the heating
    rate density will be equal to :math:`\epsilon_X = f_{X,h} c_X f_X \times \text{SFR}`.

    Default: :math:`3.4 \times 10^{40}` [:math:`\text{erg} \ \text{s}^{-1} \ (M_{\odot} \ \mathrm{yr}^{-1})^{-1}`]
    
``fX``
    Constant multiplicative factor applied to ``cX``, which is typically chosen to match observations of nearby star-forming galaxies, i.e., ``fX`` parameterizes ignorance in redshift evolution of ``cX``.
    
    Default: 1

Radiation at energies below the Lyman Limit
-------------------------------------------

``Nlw``
    Number of photons emitted in the Lyman-Werner band per baryon of star formation.
    
    Default: 9690
    
Ultraviolet emission
--------------------
``Nion``
    Number of ionizing photons emitted per baryon of star formation.
    
    Default: 4000
    
``fesc``
    Escape fraction of ionizing radiation.
    
    Default: 0.1

Not done yet
------------

::

    "source_type": 'star',
    "source_kwargs": None,
    
    "model": -1, # Only BHs use this at this point
    
    "zoff": 5.0,
    
    # Bypass fcoll prescriptions, use parameterizations
    "emissivity": None,
    "epsilon_X": None,
    "Gamma_HI": None,
    "gamma_HI": None,
    
    "xray_Eavg": 500.,
    "uv_Eavg": 30.,
                

    