The Intergalactic Medium
========================
We've seen so far that you can initialize stellar and BH populations and 
radiation backgrounds independently, i.e., without doing a full blown 
21-cm calculation. However,
the radiation background will in general modify the 
properties of the intergalactic medium, which will then influence the subsequent
evolution of the radiation background (and so on). Coupling radiation from
stars and BHs to the IGM requires use of the ``glorb.evolve.IntergalacticMedium`` 
module:

::

    import glorb
    
    igm = glorb.evolve.IGM()
    
    # By default, assumes IGM is neutral
    tau_neutral = igm.OpticalDepth(10, 12, 500)

    # Can supply ionized fraction as constant
    tau_xconst = igm.OpticalDepth(10, 12, 500, xavg=0.5)
    
    # Or, supply ionized fraction as function of redshift - how about a tanh model?
    from glorb.util import xHII_tanh
    
    xavg = lambda z: xHII_tanh(z, zr=10, dz=2)
    
    tau_xtanh = igm.OpticalDepth(10, 12, 500, xavg=xavg)

In order to compute the ionization and heating rate in the IGM with time, we 
need to know something about the radiation background.

