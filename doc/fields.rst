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

* Rate coefficient for photo-ionization ``Gamma_h_1``.
* etc.

Two-Zone IGM Models
-------------------
For calculations of the reionization history or global 21-cm signal, in which we use a two-zone IGM formalism, all quantities described in the previous sections keep their usual names with one important change: they now also have an `igm` or `cgm` prefix to signify which phase of the IGM they belong to. The `igm` phase is of course short for inter-galactic medium, while the `cgm` phase stands for the circum-galactic medium (really just meant to indicate gas near galaxies).

* Kinetic temperature, ``igm_Tk``.
* HII region volume filling factor, ``cgm_h_2``.
* Neutral fraction in the bulk IGM, ``igm_h_1``.
* Heating rate in the IGM, ``igm_heat_h_1``.
* Volume-averaged ionization rate, or rate of change in ``cgm_h_2``, ``cgm_Gamma_h_1``.


There are also new (passive) quantities, like the neutral hydrogen excitation
(or ``spin'' temperature), the 21-cm brightness temperature, and rate
coefficients that govern the time evolution of the 21-cm signal:

* 21-cm brightness temperature: ``'igm_dTb'``.
* Spin temperature: ``'igm_Ts'``.

Each of these are only associated with the IGM grid patch, since the other phase of the IGM is assumed to be fully ionized and thus dark at 21-cm wavelengths.



