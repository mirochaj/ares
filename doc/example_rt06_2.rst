RT06 Test #2 (Strömgren Sphere, thermal evolution allowed)
==========================================================
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

:: 

    import ares
    
    sim = ares.simulations.RaySegment(problem_type=2)
    sim.run()
    
    anl = ares.analysis.RaySegment(sim.checkpoints)
    
    anl.PlotIonizationFrontEvolution(fig=1)

    # Snapshots at 10 and 50 Myr
    anl.IonizationProfile(fig=2, t=[10, 50])
    anl.TemperatureProfile(fig=3, t=[10, 50])
    
    
    