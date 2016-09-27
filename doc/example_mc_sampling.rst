Monte-Carlo Sampling Higher Dimensional Spaces
==============================================
For one- or two-dimensional parameter studies, a gridded search of parameter space (as in :doc:`example_grid`) is a reasonable approach. However, as the dimensionality grows, things quickly get out of hand. As a result, it can sometimes be advantageous to run Monte Carlo simulations instead, sampling more sparsely (but more efficiently) a high-dimensional space.

You can do this in *ares* using the ``ModelSample`` class, which is just a wrapper around the ``ModelGrid`` class. As a result, the problem setup is very similar to that in :doc:`example_grid`, and that structure of the output data are identical, which means the routines documented in :doc:`example_grid_analysis` translate as well.

Before we start, the few usual imports:

::

    import ares
    import numpy as np
    
Our "go-to" Efficient Example: :math:`tanh` model for the global 21-cm signal
-----------------------------------------------------------------------------
To facilitate a comparison between the model grid results, let's start by choosing the same blobs as in :doc:`example_grid`:

::

    blobs_scalar = ['z_D', 'dTb_D', 'tau_e']
    blobs_1d = ['cgm_h_2', 'igm_Tk', 'dTb']
    blobs_1d_z = np.arange(5, 21)
        
    base_pars = \
    {
     'problem_type': 101,
     'tanh_model': True,
     'blob_names': [blobs_scalar, blobs_1d],
     'blob_ivars': [None, blobs_1d_z],
     'blob_funcs': None,
    }
    
Now, instead of creating a ``ModelGrid`` instance, we make a ``ModelSample`` instance:
    
::

    mc = ares.inference.ModelSample(**base_pars)
    
At this point we have yet to specify which parameters will to sample. Because we are now doing Monte Carlo simulations, we must define the *distributions* from which to draw samples in each parameter of interest, rather than the grid of values to sample. To do this we use the ``PriorSet``, which is important to :doc:`example_mcmc_I` as well:

::

    from ares.inference.PriorSet import PriorSet

    ps = PriorSet()
    
Now, let's study the same parameters as :doc:`example_grid` with one addition: the duration of "reheating":

::

    from ares.inference.Priors import UniformPrior

    # Draw samples from a uniform distribution between supplied (min, max) values for each parameter
    ps.add_prior(UniformPrior(6, 12), 'tanh_xz0')
    ps.add_prior(UniformPrior(0.1, 8), 'tanh_xdz')
    ps.add_prior(UniformPrior(0.1, 8), 'tanh_Tdz')
    
    # Give distributions to the ModelSample instance
    mc.prior_set = ps

.. note :: You can also draw samples from a Gaussian (via ``GaussianPrior``), a truncated Gaussian (``TruncatedGaussianPrior``), and many more. See ares.inference.Priors for a complete listing.

One last thing: we must specify how many random samples to draw:

::

    mc.N = 2e3          # Number of models to run    
    
Finally, to run it:

::

    mc.run('test_3d_mc', clobber=True, save_freq=100)

To analyze the results, create an analysis instance,    

::

    anl = ares.analysis.ModelSet('test_3d_mc')
    
and, for example, plot the 2-d parameter space with points color-coded by ``tau_e``

::

    anl.Scatter(['tanh_xz0', 'tanh_xdz'], c='tau_e', edgecolors='none')
    
Now that we have also varied the thermal history through ``tanh_Tdz``, we can look at the interplay between reionization and reheating in setting the emission maximum of the global signal, e.g., 

::

    anl.Scatter(['tanh_xdz', 'tanh_Tdz'], c='dTb_D', edgecolors='none', fig=2)
    
See :doc:`example_grid_analysis` for more information.

