:orphan:

Analyzing Model Grids
=====================
Once you have a model grid in hand, there are a slew of built-in analysis 
routines that you might consider using. For the rest of this example,
we'll assume you have completed a :doc:`example_grid`, and have the associated set of files
with prefix ``test_2d_grid``. If not, you can download a tarball of that model grid `here <https://bitbucket.org/mirochaj/ares/downloads/ares_example_grid.tar.gz>`_.

To begin, initialize an instance of the analysis class, ``ModelSet``: 

::

    import ares

    anl = ares.analysis.ModelSet('test_2d_grid')

First, let's verify that we've surveyed the part of parameter space we 
intended to: 

::

    ax = anl.Scatter(anl.parameters)
    
You should see a scatterplot with points in the :math:`z_{\mathrm{rei}}`---:math:`\Delta z_{\mathrm{rei}}` 
plane representing the models in our grid.

Basic Inspection
----------------
Now, the kind of analysis you can do will be limited by what quantities
were saved for each model. Recall (from :doc:`example_grid`) that we have 
the following quantities at our disposal:

* 21-cm brightness temperature, ``dTb``.
* Kinetic temperature of the IGM, ``igm_Tk``.
* HII region volume filling factor, ``cgm_h_2``.
* CMB optical depth, ``tau_e``.
* Position of 21-cm emission maximum, ``z_D`` and ``dTb_D``.

Let's have a look at how the ionization history depends on our two parameters of
interest in this example, ``tanh_xz0`` and ``tanh_xdz``,

::

    ax = anl.Scatter(['tanh_xz0', 'tanh_xdz'], c='cgm_h_2', ivar=[None,None,10], fig=2, edgecolors='none')

.. note :: All ``ModelSet`` functions accept ``fig`` as an optional keyword 
    argument, which you can set to any integer to open plots in a new window.    

The keyword argument ``ivar`` is short for "independent variables" -- it is ``None`` by default. However, because we have chosen to plot ``cgm_h_2``, which is a 1-D blob, we must specify the redshift of interest. Recall that we have access to integer redshifts in the interval :math:`5 \leq z \leq 20`, or check for yourself:

::
    
    print anl.blob_ivars
    
So our choice of :math:`z=10` should be OK.  

If you forget what fields are available for analysis, see:

::

    print anl.blob_names, anl.blob_ivars

.. note :: Calling quantities of interest `blobs` was inspired by the arbitrary meta-data blobs in `emcee <http://dan.iel.fm/emcee/current/>`_.   
 
For more 21-cm-focused analyses, you may want to view how the extrema in the
global 21-cm signal change as a function of the model parameters:

::
    
    # Scatterplot showing where emission peak occurs
    ax = anl.Scatter(['z_D', 'dTb_D'], fig=4)

or, color-code points by CMB optical depth,

::

    ax = anl.Scatter(['z_D', 'dTb_D'], c='tau_e', fig=5, edgecolors='none')

You can also create your own derived quantities. A very simple example is to convert redshifts to observed frequencies,

::

    # 1420.4057 is the rest frequency of the 21-cm line in MHz
    anl.DeriveBlob('1420.4057 / (1. + x)', varmap={'x': 'z_D'}, name='nu_D')
    
This will create a new blob, called ``nu_D``, that can be used for subsequent analysis. For example,

::

    # Scatterplot showing where emission peak occurs
    ax = anl.Scatter(['nu_D', 'dTb_D'], c='tau_e', fig=6, edgecolors='none')

Problem Realizations
--------------------    
You may have noticed that in this model grid there are three realizations whose emission maxima seem to occur at :math:`\delta T_b \approx 0`. In general, this is possible, but given the regularity of the grid points in parameter space it seems unlikely that any individual model would stray substantially from the locus of all other models.

To inspect potentially problematic realizations, it is first useful to isolate them from the rest. You can select them visually by first invoking

::

    anl.SelectModels()
    
and then clicking and dragging within the plot window to define a rectangle, starting from its upper left corner (click) and ending with its bottom right corner (release). The set of models bounded by this rectangle will be saved as a new ``ModelSet`` object that can be used just like the original one. Each successive "slice" will be saved as attributes ``slice_0``, ``slice_1``, etc. that you can assign to a new variable, as, e.g.

::

    slc0 = anl.slice_0
    slc0.Scatter(['nu_D', 'dTb_D'], c='tau_e', fig=7, edgecolors='none')
    
Alternatively, you can specify a rectangle by hand. For example, 

::

    slc = anl.Slice([100, 120, 0, 10], pars=['nu_D', 'dTb_D'])
    
extracts all models with ``100 <= nu_D <= 120`` and ``0 <= dTb_D <= 10``. Check:

::

    slc.Scatter(['nu_D', 'dTb_D'], c='tau_e', fig=8, edgecolors='none')
    
If you wanted to examine models in more detail, you could re-run them. Collecting the parameter dictionaries required to do so is easy:

::

    kwargs_list = slc.AssembleParametersList(include_bkw=True)
    
This routine returns a list in which each element is a dictionary of parameters for a single model. The keyword argument ``include_bkw`` controls whether the "base kwargs," i.e., those that are shared by all models in the grid, are included in each list element. If they are (as above), then any individual dictionary can be used to initialize a simulation. For example:

::
    
    ax = None
    for kwargs in kwargs_list:
        sim = ares.simulations.Global21cm(**kwargs)
        sim.run()
        ax = sim.GlobalSignature(color='b', alpha=0.5, ax=ax)
    
If you've got models that seem to have something wrong with them, sending me the dictionary (or a list of them as above) will help a lot. Just do something like:

::

    import pickle
    f = open('problematic_models.pkl', 'wb')
    pickle.dump(f)
    f.close()
    
    

.. Confidence Contours
.. -------------------
.. Notice that we have yet to assume anything about a measurement, meaning we have
.. made no attempt to quantify the likelihood that any model in our grid is 
.. correct. Let's say that somebody hands us a measurement of the position of the
.. absorption trough in the global 21-cm signal: it's at :math:`\nu=80 \pm 2` MHz and
.. :math:`\delta T_b = -100 \pm 20` mK, where the errors provided are assumed to 
.. be :math:`1−\sigma` (independent) Gaussian errors.
.. 
.. .. note :: For this example, it will be advantageous to have a more 
..     well-sampled parameter space. Consider re-running the :doc:`example_grid` 
..     with more points in each dimension before proceeding. Or, just download 
..     one `here <https://bitbucket.org/mirochaj/ares/downloads/ares_example_grid.tar.gz>`_.
.. 
.. To compute the likelihood for each model in our grid, we can define functions
.. representing the Gaussian errors on the measurement, and pass them to the
.. ``set_constraint`` function: 
.. 
.. ::
.. 
..     nuC = lambda x: np.exp(-(x - 80.)**2 / 2 / 2.**2) 
..     TC = lambda x: np.exp(-(x + 100.)**2 / 2. / 10.**2)
..     anl.set_constraint(nu=['C', nuC], dTb=['C', TC])
..     
.. Each argument passed to ``set_constraint`` is a two-element list: the redshift
..     
..     
.. Now, to look at the probability distribution function for our parameters of 
.. interest, 
.. 
.. ::
.. 
..     ax = anl.PosteriorPDF(['fX', 'fstar'], take_log=True)
.. 
.. .. note :: It may often be advantageous to supply ``take_log=True`` in order 
..     to view posterior PDFs of quantities in log-log space.
.. 
.. To convert the color-scale from one proportional to the likelihood of a given
.. model to one that denotes, e.g., the 1 and 2 :math:`\sigma` bounds on the 
.. likelihood, do something like: 
.. 
.. ::
.. 
..     ax = anl.PosteriorPDF(['fX', 'fstar'], take_log=True, color_by_like=True,
..         colors=['g', 'b'])
..         
.. By default, this includes the 68 and 95 percent confidence intervals, but you
.. can pick any contour(s) you like (no matter how unconventional it might be):
.. 
.. ::
.. 
..     ax = anl.PosteriorPDF(['fX', 'fstar'], take_log=True, color_by_like=True,
..         colors=['g', 'b'], nu=[0.5, 0.8])
..         
.. .. note :: To view the confidence regions as open contours, set 
..     ``filled=False``. You can control the color and linestyle of each contour 
..     by the ``colors`` and ``linestyles`` keyword arguments.

.. Extracting Subsets of Models
.. ----------------------------
.. Often you may want to focus on some subset of models within a grid. There
.. are a few different ways of doing this in `ares`. The model grid from above 
.. (in section on confidence contours) will make for a nice test dataset.
.. 
.. To read in that dataset, 
.. 
.. ::
.. 
..     anl = ares.analysis.ModelSet('test_grid_30x80')
.. 
.. Then, set the constraints as we did before:
.. 
.. ::
.. 
..     constraints = \
..     {
..      'nu': ['C', lambda x: np.exp(-(x - 80.)**2 / 2 / 2.**2)], 
..      'dTb': ['C', lambda x: np.exp(-(x + 100.)**2 / 2. / 10.**2)],
..     }
.. 
..     # Set constraints
..     anl.set_constraint(**constraints)
.. 
..         
.. and visualize
..     
.. ::
.. 
..     ax = anl.PosteriorPDF(['fX', 'fstar'], take_log=[True, True], 
..         color_by_like=True)
..         
.. Now, to select only the models within the :math:`2-\sigma` confidence contour 
.. in the :math:`f_X-f_{\ast}` plane, for example, we can take a *slice* through the model 
.. grid:
.. 
.. ::
.. 
..     new_anl = anl.Slice(['fX', 'fstar'], like=0.95, take_log=True, 
..         **constraints)
.. 
.. The returned value is a new instance of `ModelSet`. To convince yourself that
.. you've retrieved the correct data, overplot the ``new`` dataset as points 
.. on the previous axes (with the posterior PDF):
..         
.. ::
..         
..     new_anl.Scatter('fX', 'fstar', take_log=[True, True], 
..         ax=ax, color='r', label=r'$\mathcal{L} > 0.95$')
..     
.. You can also extract a subset of models that have some desired set of 
.. properties, independent of likelihood. For example, to extract all models 
.. with absorption troughs located at :math:`72 \leq \nu / \text{MHz} \leq 88` 
.. and :math:`-120 \leq \delta T_b / \text{mK} \leq -80`, you would do:
.. 
.. ::
..     
..     new_constraints = \
..     {
..      'nu': ['C', lambda x: 1 if 72 <= x <= 88 else 0],
..      'dTb': ['C', lambda x: 1 if -120 <= x <= -80 else 0],
..     }
..     
..     # Take slice and return new ModelSet instance
..     new_anl = anl.Slice(['fX', 'fstar'], bins=100, 
..         take_log=True, **new_constraints)
..         
..     # Overplot new points on previous axis    
..     new_anl.Scatter('fX', 'fstar', take_log=[True, True], 
..         ax=ax, color='c', facecolors='none', label='crude slice')
..     
..     ax.legend(fontsize=14)
..     pl.draw()
..     
.. 
.. Highly Dimensional Grids
.. ------------------------
.. For parameter studies with :math:`\gtrsim 3` dimensions, you might want to use 
.. MCMC. See :doc:`example_mcmc_I` for an example.
.. 
.. 
.. 