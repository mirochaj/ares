Field Listing
=============
The most fundamental quantities associated with any calculation done in ares
are the gas density, species fractions and the gas temperature. 

Species Fractions
-----------------
Our naming convention is to denote ions using their chemical symbol (in lower-case), followed by the ionization state, separated by an underscore. Rather than denoting the ionization state with roman numerals, we simply use integers. For example, neutral hydrogen is `h_1` and ionized hydrogen is `h_2`. 

Here is a complete listing:

* Neutral hydrogen fraction: ``'h_1'``
* Ionized hydrogen fraction: ``'h_2'``
* Neutral helium fraction: ``'he_1'`` 
* Singly-ionized helium fraction: ``'he_2'``
* Doubly-ionized helium fraction: ``'he_3'``
* Electron fraction: ``'e'``
* Gas density (in :math:`g \ \text{cm}^{-3}`): ``'rho'``

These are the default elements in the ``history`` dictionary, which is an attribute of all ``ares.simulations`` classes. 

We also generally keep track of the ionization and heating rate coefficients:

* Rate coefficient for photo-ionization, ``k_ion``.
* Rate coefficient for secondary ionization by photo-electrons, ``k_ion2``.
* Rate coefficient for photo-heating, ``k_heat``.

Each of these quantities are multi-dimensional because we store the rate coefficients for each absorbing species separately. 

Two-Zone IGM Models
-------------------
For calculations of the reionization history or global 21-cm signal, in which we use a two-zone IGM formalism, all quantities described in the previous sections keep their usual names with one important change: they now also have an `igm` or `cgm` prefix to signify which phase of the IGM they belong to. The `igm` phase is of course short for inter-galactic medium, while the `cgm` phase stands for the circum-galactic medium (really just meant to indicate gas near galaxies).

* Kinetic temperature, ``igm_Tk``.
* HII region volume filling factor, ``cgm_h_2``.
* Neutral fraction in the bulk IGM, ``igm_h_1``.
* Heating rate in the IGM, ``igm_k_heat``.
* Volume-averaged ionization rate, ``cgm_k_ion``.

There are also new (passive) quantities, like the neutral hydrogen excitation
(or ``spin'' temperature), the 21-cm brightness temperature, and the Lyman-:math:`\alpha` background intensity:

* 21-cm brightness temperature: ``'igm_dTb'``.
* Spin temperature: ``'igm_Ts'``.
* :math:`J_{\alpha}`: ``'igm_Ja'``.

Each of these are only associated with the IGM grid patch, since the other phase of the IGM is assumed to be fully ionized and thus dark at 21-cm wavelengths.



