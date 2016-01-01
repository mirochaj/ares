:orphan:

Spectrum Parameters
===================
We use :math:`I_{\nu}` to denote to the SED of sources. It is proportional
to the *energy* emitted at frequency :math:`\nu`, NOT the number of photons
emitted at frequency :math:`\nu`, and is normalized such that

.. math::

    \int_{\text{spectrum_EminNorm}}^{\text{spectrum_EminNorm}} I_{\nu} d\nu = 1

We use square brackets on this page to denote the units of parameters.

``spectrum_type``
    Options:

    + ``'bb'``: blackbody
    + ``'pl'``: power-law
    + ``'mcd'``; Multi-color disk (Mitsuda et al. 1984)
    + ``'simpl'``: SIMPL Comptonization model (Steiner et al. 2009)
    + ``'qso'``: Quasar template spectrum (Sazonov et al. 2004)

``spectrum_Emin``
    Minimum photon energy to consider in radiative transfer calculation.

    Default: 200 [eV]

``spectrum_Emax``
    Maximum photon energy to consider in radiative transfer calculation. 

    Default: 3e4 [eV]

``spectrum_EminNorm``
    Minimum photon energy to consider in normalization.
    
    Default: 200 [eV]

``spectrum_EmaxNorm``
    Maximum photon energy to consider in normalization.

    Default: 3e4 [eV]
    
``spectrum_alpha``
    Power-law index of emission. Only used if ``spectrum_type`` is ``pl`` or ``simpl``. Defined such that :math:`I_{\nu} \propto \nu^{\alpha}`.
    
    Default: -1.5
    
Recall that :math:`I_{\nu}` is proportional to the energy, not the number of photons,
emitted at frequency :math:`\nu`.
    
``spectrum_logN``
    Base-10 logarithm of the neutral absorbing column in units of :math:`\text{cm}^{-2}`.
    
    Default: :math:`-\infty`
    
``spectrum_Rmax``
    If ``spectrum_type`` is 'mcd', this parameter sets the maximum size of the
    accretion disk being considered.
    
    Default: 1000 [gravitational radii, :math:`R_g = G M_{\bullet} / c^2`, where :math:`M_{\bullet}` is the black hole mass]
    
``spectrum_fcol``
    Color correction factor, acts to harden BH accretion spectrum. 
    
    Default: 1.7
    
``spectrum_kwargs``
    A dictionary containing any (or all) spectrum parameters.
    
