Analyzing Model Grids
=====================
Once you have a model grid in hand, there are a slew of built-in analysis 
routines that you might consider using. For the rest of this example,
we'll assume you have completed a :doc:`example_grid_I`, and have the associated set of files
with prefix ``test_model_grid``.

To begin, initialize an instance of the analysis class, ``ModelSet``: ::

    anl = ares.analysis.ModelSet('test_model_grid')

First, let's verify that we've surveyed the part of parameter space we 
intended to: ::

    ax = anl.Scatter('fX', 'fstar')
    
You should see a scatterplot with points in the :math:`f_X`---:math:`f_{\ast}` 
plane representing the models in our grid.

Basic Inspection
----------------
Now, the kind of analysis you can do will be limited by what quantities
were saved for each model. Recall (from :doc:`example_grid_I`) that we have 
the following quantities at our disposal:

* 21-cm brightness temperature, ``dTb``.
* Spin temperature, ``Ts``.
* Kinetic temperature, ``igm_Tk``.
* HII region volume filling factor, ``cgm_h_2``.
* Neutral fraction in the bulk IGM, ``igm_h_1``.
* Heating rate in the IGM, ``igm_heat_h_1``.
* Volume-averaged ionization rate, i.e., the rate of change in ``cgm_h_2``, ``cgm_Gamma_h_1``.

and we know them at redshifts 6, 8, 10, 12, and the redshifts corresponding 
to turning points B, C, and D in the global 21-cm signal. 

Let's have a look at how the thermal history depends on our two parameters of
interest in this example, :math:`f_X` and :math:`f_{\ast}`. 

::

    ax = anl.Scatter('fX', 'igm_Tk', z=10, fig=2)

.. note :: All ``ModelSet`` functions accept ``fig`` as an optional keyword argument, which you can set to any integer to open plots in a new window.    

Notice that there are two tracks of points, one for each value of :math:`f_{\ast}`.
This information can be visualized on the same axis by supplying the keyword
argument ``c``, which color-codes each point by the field provided:

::

    ax = anl.Scatter('fX', 'igm_Tk', c='fstar', z=10, fig=3)

For more 21-cm-focused analyses, you may want to view how the extrema in the
global 21-cm signal change as a function of the model parameters:

::
    
    # Scatterplot showing where absorption/emission peaks occur
    ax = anl.Scatter(x='nu', y='dTb', z='C', fig=4)
    ax = anl.Scatter(x='nu', y='dTb', z='D', ax=ax)
    
    # Run default global 21-cm signal model
    sim = ares.simulations.Global21cm()
    sim.run()
    
    # Overplot it
    anl_sim = ares.analysis.Global21cm(sim)
    anl_sim.GlobalSignature(ax=ax)
    
If you forget what fields are available for analysis (and at what redshifts),
see:

::

    print anl.blob_names, anl.blob_redshifts
    
.. note :: Calling quantities of interest `blobs` was inspired by the arbitrary meta-data blobs in `emcee <http://dan.iel.fm/emcee/current/>`_. 

Confidence Contours
-------------------
Notice that we have yet to assume anything about a measurement, meaning we have
made no attempt to quantify the likelihood that any model in our grid is 
correct. Let's say that somebody hands us a measurement of the position of the
absorption trough in the global 21-cm signal: it's at :math:`\nu=80 \pm 2` MHz and
:math:`\delta T_b = -100 \pm 20` mK, where the errors provided are assumed to 
be :math:`1−\sigma` (independent) Gaussian errors.

.. note :: For this example, it will be advantageous to have a more 
    well-sampled parameter space. Consider re-running the :doc:`example_grid_I` 
    with more points in each dimension before proceeding. Or, just download 
    one `here <https://bitbucket.org/mirochaj/ares/downloads/ares_example_grid.tar.gz>`_.

To compute the likelihood for each model in our grid, we can define functions
representing the Gaussian errors on the measurement, and pass them to the
``set_constraint`` function: 

::

    nuC = lambda x: np.exp(-(x - 80.)**2 / 2 / 2.**2) 
    TC = lambda x: np.exp(-(x + 100.)**2 / 2. / 10.**2)
    anl.set_constraint(nu=['C', nuC], dTb=['C', TC])
    
Each argument passed to ``set_constraint`` is a two-element list: the redshift
    
    
Now, to look at the probability distribution function for our parameters of 
interest, 

::

    ax = anl.PosteriorPDF(['fX', 'fstar'], take_log=True)

.. note :: It may often be advantageous to supply ``take_log=True`` in order 
    to view posterior PDFs of quantities in log-log space.

To convert the color-scale from one proportional to the likelihood of a given
model to one that denotes, e.g., the 1 and 2 :math:`\sigma` bounds on the 
likelihood, do something like: 

::

    ax = anl.PosteriorPDF(['fX', 'fstar'], take_log=True, color_by_like=True,
        colors=['g', 'b'])
        
By default, this includes the 68 and 95 percent confidence intervals, but you
can pick any contour(s) you like (no matter how unconventional it might be):

::

    ax = anl.PosteriorPDF(['fX', 'fstar'], take_log=True, color_by_like=True,
        colors=['g', 'b'], nu=[0.5, 0.8])
        
.. note :: To view the confidence regions as open contours, set 
    ``filled=False``. You can control the color and linestyle of each contour 
    by the ``colors`` and ``linestyles`` keyword arguments.

Extracting Subsets of Models
----------------------------
Often you may want to focus on some subset of models within a grid. There
are a few different ways of doing this in `ares`. The model grid from above 
(in section on confidence contours) will make for a nice test dataset.

To read in that dataset, 

::

    anl = ares.analysis.ModelSet('test_grid_30x80')

Then, set the constraints as we did before:

::

    constraints = \
    {
     'nu': ['C', lambda x: np.exp(-(x - 80.)**2 / 2 / 2.**2)], 
     'dTb': ['C', lambda x: np.exp(-(x + 100.)**2 / 2. / 10.**2)],
    }

    # Set constraints
    anl.set_constraint(**constraints)

        
and visualize
    
::

    ax = anl.PosteriorPDF(['fX', 'fstar'], take_log=[True, True], 
        color_by_like=True)
        
Now, to select only the models within the :math:`2-\sigma` confidence contour 
in the :math:`f_X-f_{\ast}` plane, for example, we can take a *slice* through the model 
grid:

::

    new_anl = anl.Slice(['fX', 'fstar'], like=0.95, take_log=True, 
        **constraints)

The returned value is a new instance of `ModelSet`. To convince yourself that
you've retrieved the correct data, overplot the ``new`` dataset as points 
on the previous axes (with the posterior PDF):
        
::
        
    new_anl.Scatter('fX', 'fstar', take_log=[True, True], 
        ax=ax, color='r', label=r'$\mathcal{L} > 0.95$')
    
You can also extract a subset of models that have some desired set of 
properties, independent of likelihood. For example, to extract all models 
with absorption troughs located at :math:`72 \leq \nu / \text{MHz} \leq 88` 
and :math:`-120 \leq \delta T_b / \text{mK} \leq -80`, you would do:

::
    
    new_constraints = \
    {
     'nu': ['C', lambda x: 1 if 72 <= x <= 88 else 0],
     'dTb': ['C', lambda x: 1 if -120 <= x <= -80 else 0],
    }
    
    # Take slice and return new ModelSet instance
    new_anl = anl.Slice(['fX', 'fstar'], bins=100, 
        take_log=True, **new_constraints)
        
    # Overplot new points on previous axis    
    new_anl.Scatter('fX', 'fstar', take_log=[True, True], 
        ax=ax, color='c', facecolors='none', label='crude slice')
    
    ax.legend(fontsize=14)
    pl.draw()
    

Highly Dimensional Grids
------------------------
For parameter studies with :math:`\gtrsim 3` dimensions, you might want to use 
MCMC. See :doc:`example_mcmc_I` for an example.


