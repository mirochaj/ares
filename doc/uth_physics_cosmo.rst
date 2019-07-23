:orphan:

Cosmology
=========
One quick note about cosmology: the convention in *ares* is to eliminate all factors of "little h," i.e., the Hubble parameter in units of :math:`100 \ \mathrm{km} \ \mathrm{s}^{-1} \ \mathrm{Mpc}^{-3}`. The most noticeable place where this happens is in the ``ares.physics.HaloMassFunction`` class. For example, whereas the *hmf* code yields the halo mass function with implicit h's, *ares* "undoes" these factors, meaning, e.g., that the halo mass function stored in ``ares.physics.HaloMassFunction.tab_dndm`` is simply in units of :math:`\mathrm{Mpc}^{-3}`, not :math:`h^4 \mathrm{Mpc}^{-3}`, so the user need not multiply :math:`tab_dndm` by :math:`h^4` to obtain the "true" mass function. The same goes for halo masses themselves (no need to divide by :math:`h`) and the cumulative mass function (no need to multiply by :math:`h^3` or :math:`h^2` for :math:`n(>m)` and :math:`m(>m)`, respectively).

For a nice discussion of little h check out `this paper by Darren Croton <https://arxiv.org/abs/1308.4150>`_.