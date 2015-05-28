Cosmological Radiative Transfer
===============================
We've seen so far that you can initialize stellar and BH populations and 
radiation backgrounds independently, i.e., without doing a full blown 
21-cm calculation. However,
the radiation background will in general modify the 
properties of the intergalactic medium, which will then influence the subsequent
evolution of the radiation background (and so on). Coupling radiation from
stars and BHs to the IGM requires use of the ``ares.solvers.IntergalacticMedium`` 
module:

::

    import ares
    
    igm = ares.solvers.IGM()
    
    # Optical depth between 10 <= z <= 12 at 500 eV. 
    # By default, assumes IGM is neutral
    tau_neutral = igm.OpticalDepth(10, 12, 500)

    # Can supply ionized fraction as constant
    tau_xconst = igm.OpticalDepth(10, 12, 500, xavg=0.5)
