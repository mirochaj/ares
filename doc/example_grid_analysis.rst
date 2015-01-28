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
    

Confidence Contours
-------------------
Notice that we have yet to assume anything about a measurement, meaning we have
made no attempt to quantify the likelihood that any model in our grid is 
correct. Let's say that somebody hands us a measurement of the IGM temperature
at :math:`z=10`: it is :math:`T_k(z=10) = 400 \pm 20` Kelvin.

.. note :: For this example, it will be advantageous to have a more well-sampled parameter space. Consider re-running the :doc:`example_grid_I` with more points in each dimension before proceeding.

Under construction sorry!

