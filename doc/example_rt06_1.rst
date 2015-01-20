RT06 Test #1 (Strömgren Sphere, isothermal)
============================================
Test #1 from the Radiative Transfer Comparison Project (`Iliev et al. 2006 <http://adsabs.harvard.edu/abs/2006MNRAS.371.1057I>`_).

This problem investigates the growth of an HII region around a monochromatic 
source of ionizing photons. The main parameters are:

* Stellar ionizing photon production rate of :math:`\dot{Q} = 5 \times 10^{48} \ \text{s}^{-1}`. 
* Medium composed of hydrogen only, with a density of :math:`n_{\text{H}} = 10^{-3} \ \text{cm}^{-3}`.
* Medium is isothermal at :math:`T=10^4` K.

:: 

    import ares
    
    sim = ares.simulations.RaySegment(problem_type=1)
    sim.run()
    
    anl = ares.analysis.RaySegment(sim.checkpoints)
    anl.PlotIonizationFrontEvolution()
