Population Parameters
=====================



Timing etc.
-----------
``formation_epoch``
    Redshift interval over which sources are "on."

    Default: (50, 0)
    
``is_lya_src`` 
    Sources contribute to Ly-:math:`\alpha` background?
    
    Default: ``True``

``is_ion_src_cgm`` 
    Sources contribute to growth of HII regions?

    Default: ``True``

``is_ion_src_igm`` 
    Sources contribute ionization in bulk IGM?
    
    If ``approx_xrb=True``, this ionization rate a mean X-ray photon energy
    of ``xray_Eavg``, which is 500 eV by default.

    Default: ``True``
    
``is_heat_src_igm``
    Sources emit X-rays and heat bulk IGM?
    
    Default: ``True``
    
``solve_rte``
    Solve the cosmological radiative transfer equation (RTE) in detail?
    
    Options: bool, list, tuple
    
    Default: ``False``
    
Star formation history
----------------------    
    
``pop_Tmin``
    Minimum virial temperature of star-forming halos.
    
    Default: :math:`10^4` [Kelvin]
    
``pop_Mmin``
    Minimum mass of star-forming halos. Will override ``Tmin`` if set to 
    something other than ``None``.

    Default: ``None`` [:math:`M_{\odot}`]

``pop_fstar``
    Star formation efficiency, :math:`f_{\ast}`, i.e., fraction of collapsing
    gas that turns into stars.
    
    Default: 0.1

``pop_sfrd``
    The star formation rate density (SFRD) as a function of redshift. If provided, will override ``Tmin`` and ``Mmin``. For example, a constant (co-moving) SFRD of :math:`1 \ M_{\odot} \ \text{yr}^{-1} \ \text{cMpc}^{-3}` would be ``sfrd=lambda z: 1.0``.
    
    Default: ``None`` [:math:`M_{\odot} \ \text{yr}^{-1} \ \text{cMpc}^{-3}`]
        
Radiation Fields
----------------
``pop_yield``
    How many photons are emitted per unit star formation?
    
    Default: :math:`2.6 \times 10^{39}`
    
``pop_yield_units``
    How to normalize the yield? 
    
    Options: ``erg/s/SFR`` [i.e., :math:`\mathrm{erg} \ \mathrm{s}^{-1} \ (M_{\odot} \ \mathrm{yr}^{-1})^{-1}`], ``photons/baryon``, ``photons/Msun``
        
    Default: ``erg/s/SFR``
    
Internally, all units are cgs, which means at run-time all yields will be converted to units of :math:`\mathrm{erg} \ \mathrm{g}^{-1}`.

These parameters of course dictate an amount of energy produced per unit star formation *in a particular band*. That band is specified by the ``pop_EminNorm`` and ``pop_EmaxNorm`` parameters.

``pop_EminNorm``
    Minimum photon energy to consider in normalization.
    
    Default: 200 [eV]

``pop_EmaxNorm``
    Maximum photon energy to consider in normalization.

    Default: 3e4 [eV]
    
To be precise,

.. math ::

    \int_{\texttt{pop_EminNorm}}^{\texttt{pop_EmaxNorm}} \frac{\epsilon_{\nu}}{\dot{\rho}_{\ast}} d\nu = \frac{\texttt{pop_yield}}{\texttt{pop_yield_units}}
    
where :math:`\epsilon_{\nu}` is the emissivity of the population and :math:`\dot{\rho}_{\ast}` is the star-formation rate density (SFRD).

This range does not necessarily determine the band in which photons are emitted. For example, you might want to normalize the emission in the 0.5-8 keV band (e.g., if you're adopting the :math:`L_X`-SFR relation), but allow sources to emit at all energies. To do so, you must choose an SED, which then gets used to extrapolate the 0.5-8 keV yield to lower/higher energies.

We use square brackets on this page to denote the units of parameters.

``pop_type``
    Options:

    + ``'bb'``: blackbody
    + ``'pl'``: power-law
    + ``'mcd'``; Multi-color disk (Mitsuda et al. 1984)
    + ``'simpl'``: SIMPL Comptonization model (Steiner et al. 2009)
    + ``'qso'``: Quasar template spectrum (Sazonov et al. 2004)

``pop_Emin``
    Minimum photon energy to consider in radiative transfer calculation.

    Default: 200 [eV]

``pop_Emax``
    Maximum photon energy to consider in radiative transfer calculation. 

    Default: 3e4 [eV]
        

For backward compatibility
--------------------------
There are many parameters that do *not* have the ``pop_`` prefix attached to them, but are nonetheless convenient because they are the most common parameters in fiducial global 21-cm models. In addition, in *ares* version 0.1, the ``pop_`` formulation was not yet in place, and the following parameters were the norm. They can still be used for ``problem_type=100`` (see :doc:`problem_types`), but one should be careful otherwise.

``cX``
    Normalization of the X-ray luminosity to star formation rate (:math:`L_X`-SFR) relation in 
    band given by ``spectrum_EminNorm`` and ``spectrum_EmaxNorm``. If ``approx_xrb=1``, this
    represents the X-ray luminosity density per unit star formation, such that the heating
    rate density will be equal to :math:`\epsilon_X = f_{X,h} c_X f_X \times \text{SFR}`.

    Default: :math:`3.4 \times 10^{40}` [:math:`\text{erg} \ \text{s}^{-1} \ (M_{\odot} \ \mathrm{yr}^{-1})^{-1}`]
    
``fX``
    Constant multiplicative factor applied to ``cX``, which is typically chosen to match observations of nearby star-forming galaxies, i.e., ``fX`` parameterizes ignorance in redshift evolution of ``cX``.
    
    Default: 0.2

``Nlw``
    Number of photons emitted in the Lyman-Werner band per baryon of star formation.
    
    Default: 9690
    
``Nion``
    Number of ionizing photons emitted per baryon of star formation.
    
    Default: 4000
    
``fesc``
    Escape fraction of ionizing radiation.
    
    Default: 0.1

