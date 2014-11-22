Physics Parameters
==================

``radiative_transfer``
    0) No astrophysical sources.
    1) Includes ionization/heating from astrophysical sources.

    Default: 1

``compton_scattering``
    0) OFF
    1) Include Compton scattering between free electrons and CMB.
    
    Default: 1

``secondary_ionization``
    Determine what fraction of photo-electron energy gets deposited in various
    channels, such as heat, ionization, and excitation.
    
    0) All photo-electron energy deposited as heat.
    1) Compute using fits of Shull & vanSteenberg (1985).
    2) Compute using energy-dependent fits of Ricotti, Gnedin, & Shull (2002).
    3) Compute using look-up tables of Furlanetto & Stoever (2010).
    
    Default: 1
    
``fXh``
    Set fractional heating by photo-electrons by-hand. Currently must be a
    constant. Will override choice of ``secondary_ionization`` if supplied.
    
    Default: ``None``

``clumping_factor``
    Multiplicative enhancement to the recombination rate.
    
    Default: 1

``approx_helium``
    0) If ``Z=[1]``, neglect helium entirely. If ``Z[1,2]``, treat helium self-consistently.
    1) (Only applies if ``Z=[1,2]``) Include opacity of intergalactic helium, assuming singly-ionized helium fraction is equal to the hydrogen ionized fraction.
    
    Default: 0
    
``approx_sigma``
    0) Compute bound-free absorption cross sections via fits of Verner et al. (1996).
    1) Approximate cross-sections as :math:`\sigma \propto \nu^{-3}`
    
    Default: 0

``approx_lya``
    0) Solves RTE (i.e., full ``sawtooth'' background).
    1) Assume flat spectrum between Lyman-:math:`\alpha` and the Lyman limit.
    
    Default: 1
    
``approx_xray``
    0) Solves RTE.
    1) Heating due to instantaneous X-ray luminosity.

    Default: 1
    
``approx_Salpha``
    0) Not implemented
    1) Assume :math:`S_{\alpha} = 1`
    2) Use formulae of Chuzhoy, Alvarez, & Shapiro (2005).
    3) Use formulae of Furlanetto & Pritchard (2006)
    
    Default: 1    
    
``nmax``
    Default: 23
    
``lya_injected``
    Include photons injected at line-center?
    
    Default: ``True``    
    
``lya_continuum``
    Include photons redshifting into the blue-wing of the Lyman-:math:`\alpha` line?
    
    Default: ``True``
    
``recombination``
    Which recombination method to use? Can be ``"A"``, ``"B"``, or ``0``, the 
    first two options being standard case-A or case-B recombination, whereas
    the last option artificially turns off recombinations (useful for analytic
    tests).
    
    Default: ``B``
        