RT06 Test #2 (Strömgren Sphere, thermal evolution allowed)
==========================================================
Test #2 from the Radiative Transfer Comparison Project (`Iliev et al. 2006
<http://adsabs.harvard.edu/abs/2006MNRAS.371.1057I>`_).

This problem investigates the growth of an HII region around a star with a
surface temperature of :math:`10^5` K, whose ionizing photon luminosity is
:math:`\dot{Q} = 5 \times 10^{48} \ \text{s}^{-1}` (as in RT06 #1). The medium
is composed of hydrogen, with a density of :math:`n_{\text{H}} = 10^{-3} \
\text{cm}^{-3}`, and as opposed to RT06 #1, the gas temperature is allowed to
vary. The initial temperature is constant in space at :math:`T=100` K.

The ionization and heating rates are computed treating the source's spectral
energy distribution in full. A lengthy discussion of this can be found in
`Mirocha et al. (2012) <http://adsabs.harvard.edu/abs/2012ApJ...756...94M>`_.

:: 

    import ares
    
    sim = ares.simulations.RaySegment((problem_type=2)
    sim.run()
    
    anl = ares.analysis.RaySegment((sim.checkpoints)
    
    anl.PlotIonizationFrontEvolution()

    anl.IonizationProfile(t=[10, 50])
    
    anl.TemperatureProfile(t=[10, 50])
    
    
    