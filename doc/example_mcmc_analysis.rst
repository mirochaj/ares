:orphan:

Analyzing MCMC Calculations
===========================
Point: triangle plots, recovered quantities, etc.

If you don't yet have a dataset to work with, you can make one by following the



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



Specialized Analysis: Global 21-cm Signal
-----------------------------------------



