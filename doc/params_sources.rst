:orphan:

Source \& Spectrum Parameters
==============================
Below, we list the parameters that govern the bolometric luminosity of sources as well as their spectral energy distribution. These parameters are relevant to point sources (via the ``source_*`` parameters), as well as populations of sources used in global 21-cm signal and/or meta-galactic radiation background calculations (simply replace ``source_`` with ``pop_``).

Source Luminosity
-----------------

``source_type``
    Options:
    
    + ``star``: characterized by its temperature, ``source_temperature``, and ionizing photon luminosity, ``source_qdot``
    + ``bh``: characterized by its mass, ``source_mass`` etc.
    + ``toy``: completely parameterized

``source_temperature``
    If ``source_type`` is ``star``, this is its surface temperature. [Kelvin]

    Default: :math:`10^5 \ \text{K}`
    
``source_mass``
    If ``source_type`` is ``bh``, this is its mass [:math:`M_{\odot}`]
 
    Default: :math:`10 \ M_{\odot}` 
 
``source_qdot``
    For toy radiation source, this is the ionizing photon luminosity. [:math:`\text{s}^{-1}`]
    
    Default: :math:`5 \times 10^{48}\ \text{s}^{-1}` 
        
``source_lifetime``
    Time after which radiation from this source will no longer be considered.

    Default: :math:`10^{10}` [``time_units``]
    
The Source Spectrum
-------------------    
We'll use :math:`I_E` to denote to the SED of sources, which the user supplies via the parameter ``source_sed`` or ``pop_sed``. It is proportional
to the *energy* emitted at energy :math:`E`, NOT the number of photons
emitted at energy :math:`E`, and is normalized (automatically by *ares*) such that

.. math::

    \int_{\text{source_EminNorm}}^{\text{source_EminNorm}} \text{source_sed} dE = L_{\text{source_EminNorm}-\text{source_EminNorm}}
    
where the luminosity on the right-hand side is that determined by one of the parameters above (e.g., the Eddington luminosity of a ``source_mass`` :math:`M_{\odot}` BH) or for source populations via the ``pop_rad_yield`` parameter. This auto-normalization guarantees the radiative yield of a single source (or source population) at some photon energy is equal to its bolometric luminosity times :math:`I_E`.

We use square brackets on this page to denote the units of parameters.

``source_sed``
    Options:

    + ``'bb'``: blackbody
    + ``'pl'``: power-law
    + ``'mcd'``; Multi-color disk (Mitsuda et al. 1984)
    + ``'simpl'``: SIMPL Comptonization model (Steiner et al. 2009)
    + ``'qso'``: Quasar template spectrum (Sazonov et al. 2004)

``source_Emin``
    Minimum photon energy to consider in radiative transfer calculation.

    Default: 200 [eV]

``source_Emax``
    Maximum photon energy to consider in radiative transfer calculation. 

    Default: 3e4 [eV]

``source_EminNorm``
    Minimum photon energy to consider in normalization.

    Default: 200 [eV]

``source_EmaxNorm``
    Maximum photon energy to consider in normalization.

    Default: 3e4 [eV]

``source_alpha``
    Power-law index of emission. Only used if ``source_type`` is ``pl`` or ``simpl``. Defined such that :math:`I_{\nu} \propto \nu^{\alpha}`.

    Default: -1.5

Recall that :math:`I_{\nu}` is proportional to the energy, not the number of photons,
emitted at frequency :math:`\nu`.

``source_logN``
    Base-10 logarithm of the neutral absorbing column in units of :math:`\text{cm}^{-2}`.

    Default: :math:`-\infty`
    
``source_hardening``
    For non-zero absorbing columns, this parameter determines whether or not the 
    column is applied before or after normalizing the source's luminosity. 

    Default: ``extrinsic``
    
``source_Rmax``
    If ``source_type`` is 'mcd', this parameter sets the maximum size of the
    accretion disk being considered.

    Default: 1000 [gravitational radii, :math:`R_g = G M_{\bullet} / c^2`, where :math:`M_{\bullet}` is the black hole mass]

 
