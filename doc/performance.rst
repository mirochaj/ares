:orphan:

Performance
===========
The default parameter settings in *ARES* are not necessarily optimal for all applications. In this section, we detail some tricks that may be useful if one can tolerate a small hit in accuracy.

Time-stepping
~~~~~~~~~~~~~
For :doc:`example_gs_standard`, the main parameters that control the speed of the calculation are ``epsilon_dt`` and ``max_timestep``, which are described a bit more in :doc:`params_control`. In short, ``epsilon_dt`` determines the largest fractional change allowed in any property of the intergalactic medium (IGM) -- if a given timestep results in a larger fraction change than ``epsilon_dt`` allows, the iteration will be repeated with a smaller timestep. On the other hand, ``max_timestep`` governs the "global" timestep, i.e., the largest step we are allowed to take even in the limit where IGM quantities are evolving slowly, and thus are not restricted by ``epsilon_dt``. This parameter is in some sense aesthetic, as it determines how frequently data is saved and as a result controls the smoothness in the time evolution of quantities of interest.

The default values for these parameters, ``epsilon_dt=0.05`` and ``max_timestep=1`` (the latter in Myr) are set so that they have no discernible impact on the evolution of the IGM. However, relaxing ``epsilon_dt`` by a factor of a few and increasing ``max_timestep`` to :math:`\sim 10` Myr can provide a factor of :math:`\sim 2-3` speed-up, with only a limited impact in the results (e.g., :math:`\sim 5\%` errors induced in global 21-cm signal). Their effects have not been studied exhaustively, so it is possible that for some combinations of parameters the impact of changing these parameters may be greater. Proceed with caution!

Time-stepping is controlled a little differently in models that properly solve for the evolution of the X-ray background (as in `Mirocha (2014) <http://adsabs.harvard.edu/abs/2014arXiv1406.4120M>`_; see :doc:`example_crb_xr`). In this case, the time resolution is set to be logarithmic in :math:`1+z`, which accelerates solutions to the radiative transfer equation. The key parameter is ``tau_redshift_bins``, which is 1000 by default in the ``mirocha2017:dpl`` models (see :doc:`example_litdata`). Reducing this to 400 or 500 can result in a factor of :math:`\sim 2` speed-up. Just note that you will need to re-generate a lookup table for the IGM optical depth of that resolution -- see :doc:`inits_tables` for a few notes about how to do that (the relevant adjustment is re-setting ``Nz`` in the ``$ARES/examples/generate_optical_depth_tables.py`` script). 

Avoiding Overhead: Halo Mass Function and Stellar Population Synthesis Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Most *ARES* calculations spend :math:`\sim 10-30\%` of the run-time simply reading in some necessary look-up tables -- this sounds like a lot but is of course much faster than re-generating them on-the-fly. However, for most applications, these tables are always the same, so you can read them into memory once and pass them along to subsequent calculations for a speed-up. 

The two most common lookup tables are those for the halo mass function (HMF) and stellar population synthesis (SPS) models. The following example grabs instances of the HMF and SPS classes that are attached to an initial *ARES* simulation, and then supplies those objects to a subsequent model which then does not need to read them in for itself:

::

    import ares
    import time
    
    # First, setup a UVLF-calibrated model for the global signal.
    pars = ares.util.ParameterBundle('mirocha2017:base')
    
    # Time it
    t1 = time.time()
    sim = ares.simulations.Global21cm(**pars)
    sim.run()
    t2 = time.time()
    
    # Grab the HMF and SPS model instances from the first source population
    # and pass them to the next model via parameters that exist solely for
    # this purpose.
    pars['hmf_instance'] = sim.pops[0].halos
    pars['pop_src_instance{0}'] = sim.pops[0].src
    
    # Time the new run.
    t3 = time.time()
    sim = ares.simulations.Global21cm(**pars)
    sim.run()
    t4 = time.time()
    
    print("Sim 1 done in {} sec.".format(t2 - t1))
    print("Sim 2 done in {} sec.".format(t4 - t3))
	
This should provide a :math:`\sim 20\%` speed-up. 

.. note :: The HMF speed-up applies also to the simplest global signal models, 
	but the	``pop_src_instance`` trick used above does not, as such moddls do 
	not initialize stellar population synthesis models.

.. note :: These tricks are built-in to the ``ModelGrid`` and ``ModelFit`` 
	machinery in *ARES*. Simply set the ``save_hmf`` and ``save_psm`` attributes of each class to ``True`` before running.
	


