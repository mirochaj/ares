:orphan:

Including Helium in 1-D Radiative Transfer Calculations
=======================================================


















Test #2 from the Radiative Transfer Comparison Project (`Iliev et al. 2006
<http://adsabs.harvard.edu/abs/2006MNRAS.371.1057I>`_).

This problem investigates the growth of an HII region around a blackbody 
source of ionizing photons. The main parameters are:

* Stellar ionizing photon production rate of :math:`\dot{Q} = 5 \times 10^{48} \ \text{s}^{-1}`. 
* Stellar spectrum is a :math:`10^5` K blackbody.
* Medium composed of hydrogen only, with a density of :math:`n_{\text{H}} = 10^{-3} \ \text{cm}^{-3}`.
* Gas temperature is able to evolve. It is initially set to :math:`T=100` K everywhere on the grid.

The ionization and heating rates are computed treating the source's spectral
energy distribution in full. A lengthy discussion of this can be found in
`Mirocha et al. (2012) <http://adsabs.harvard.edu/abs/2012ApJ...756...94M>`_.

Including helium for pre-existing problem types is as simple as adding 10 to
the ``problem_type``, i.e., 

:: 

    import ares
    
    sim = ares.simulations.RaySegment(problem_type=12)
    sim.run()
    
Now, we initialize an instance of the appropriate analysis class:

::
    
    anl = ares.analysis.RaySegment(sim.checkpoints)

and have a look at the temperature profile at 10, 30, and 100 Myr,

::
    
    ax1 = anl.RadialProfile('Tk', t=[10, 30, 100])

radial profiles of the hydrogen species fractions,

::

    ax2 = anl.RadialProfile('h_1', t=[10, 30, 100], fig=2)
    anl.RadialProfile('h_2', t=[10, 30, 100], ax=ax2, ls='--')

and the species fractions for helium:

::

    ax3 = anl.RadialProfile('he_1', t=[10, 30, 100], fig=3)
    anl.RadialProfile('he_2', t=[10, 30, 100], ax=ax3, ls='--')
    anl.RadialProfile('he_3', t=[10, 30, 100], ax=ax3, ls=':')
    

    