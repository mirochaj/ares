Troubleshooting
===============
This page is an attempt to keep track of common errors and instructions for how to fix them. If you encounter a bug not listed below, `fork ares on bitbucket <https://bitbucket.org/mirochaj/ares/fork>`_ and an issue a pull request to contribute your patch, if you have one. Otherwise, shoot me an email and I can try to help. It would be useful if you can send me the dictionary of parameters for a particular calculation. For example, if you ran a global 21-cm calculation via

::

    import ares
    
    pars = {'parameter_1': 1e6, 'parameter_2': 2} # or whatever

    sim = ares.simulations.Global21cm(**pars)
    sim.run()
    
and you get weird or erroneous results, pickle the parameters:

::

    import pickle
    f = open('problematic_model.pkl', 'wb')
    pickle.dump(pars, f)
    f.close()
    
and send them to me. Thanks! 

   .. note :: If you've got a set of problematic models that you encountered            
        while running a model grid or some such thing, check out the section 
        on "problem realizations" in :doc:`example_grid_analysis`.
    

Plots not showing up
--------------------
If when running some *ARES* script the program runs to completion without errors but does not produce a figure, it may be due to your matplotlib settings. Most test scripts use ``draw`` to ultimately produce the figure because it is non-blocking and thus allows you to continue tinkering with the output if you'd like. One of two things is going on:

* You invoked the script with the standard Python interpreter (i.e., **not** iPython). Try running it with iPython, which will spit you back into an interactive session once the script is done, and thus keep the plot window open.
* Alternatively, your default ``matplotlib`` settings may have caused this. Check out your ``matplotlibrc`` file (in ``$HOME/.matplotlibrc``) and make sure ``interactive : True``. 

Future versions of *ARES* may use blocking commands to ensure that plot windows don't disappear immediately. Email me if you have strong opinions about this.

``IOError: No such file or directory``
--------------------------------------
There are a few different places in the code that will attempt to read-in lookup tables of various sorts. If you get any error that suggests a required input file has not been found, you should:

- Make sure you have set the ``$ARES`` environment variable. See the :doc:`install` page for instructions.
- Make sure the required file is where it should be, i.e., nested under ``$ARES/input``.

In the event that a required file is missing, something has gone wrong. Run ``python remote.py fresh`` to download new copies of all files.

``LinAlgError: singular matrix``
--------------------------------
This is known to occur in ``ares.physics.Hydrogen`` when using ``scipy.interpolate.interp1d`` to compute the collisional coupling coefficients for spin-exchange. It is due to a bug in LAPACK version 3.4.2 (see `this thread <https://github.com/scipy/scipy/issues/3868>`_). One solution is to install a newer version of LAPACK. Alternatively, you could use linear interpolation, instead of a spline, by passing ``interp_cc='linear'`` as a keyword argument to whatever class you're instantiating, or more permanently by adding ``interp_cc='linear'`` to your custom defaults file (see :doc:`params` section for instructions).


21-cm Extrema-Finding Not Working
---------------------------------
If the derivative of the signal is noisy (due to numerical artifacts, for example) then the extrema-finding can fail. If you can visually see three extrema in the global 21-cm signal but they are either absent or crazy in ``ares.simulations.Global21cm.turning_points``, then this might be going on. Try setting the ``smooth_derivative`` parameter to a value of 0.1 or 0.2.  This parameter will smooth the derivative with a boxcar of width :math:`\Delta z=` ``smooth_derivative`` before performing the extrema finding. Let me know if this happens (and under what circumstances), as it would be better to eliminate numerical artifacts than to smooth them out after the fact.

``AttributeError: No attribute blobs.``
---------------------------------------
This is a bit of a red herring. If you're running an MCMC fit and saving 2-D blobs, which always require you to pass the name of the function, this error occurs if you supply a function that does not exist. Check for typos and/or that the function exists where it should.

``TypeError: __init__() got an unexpected keyword argument 'assume_sorted'``
----------------------------------------------------------------------------
Turns out this parameter didn't exist prior to scipy version 0.14. If you update to scipy version >= 0.14, you should be set. If you're worried that upgrading scipy might break other codes of yours, you can also simply navigate to ``ares/physics/Hydrogen.py`` and delete each occurrence of ``assume_sorted=True``, which should have no real effect (except for perhaps a very slight slowdown).

``Failed to interpret file '<some-file>.npz' as a pickle``
----------------------------------------------------------
This is a strange one, which might arise due to differences in the Python and/or pickle version used to read/write lookup tables *ARES* uses. First, try to download new lookup tables via: ::

    python remote.py fresh
    
If that doesn't magically fix it, please email me and I'll do what I can to help!

``ERROR: Cannot generate halo mass function``
---------------------------------------------
This error generally occurs because lookup tables for the halo mass function are not being found, and when that happens, *ARES* tries to make new tables. This process is slow and so is not recommended! Instead you should check that (i) you have correctly set the $ARES environment variable and (ii) that you have run the ``remote.py`` script (see :doc:`install`), which downloads the default HMF lookup table. If you have recently pulled changes, you may need to re-run ``remote.py`` since, e.g., the default HMF parameters may have been changed and corresponding tables may have been updated on the web. To save time, you can specify that you only want new HMF tables by executing ``python remote.py fresh hmf``.


General Mysteriousness
----------------------
- If you're running *ARES* from within an iPython (or Jupyter) notebook, be wary of initializing class instances in one notebook cell and modifying attributes in a separate cell. If you re-run the the second cell *without* re-running the first cell, this can cause problems because changes to attributes will not automatically propagate back up to any parent classes (should they exist). This is known to happen (at least) when using the ``ModelGrid`` and ``ModelSamples`` classes in the inference sub-module.

