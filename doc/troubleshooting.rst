Troubleshooting
===============
This page is an attempt to keep track of common errors and instructions for how to fix them. 

Plots not showing up
--------------------
If, when running some *ares* script (e.g., those in ``$ARES/tests``) the program runs to completion without errors but does not produce a figure, it may be due to your matplotlib settings. Most test scripts use ``draw`` to ultimately produce the figure because it is non-blocking and thus allows you to continue tinkering with the output if you'd like. One of two things is going on:

* You invoked the script with the standard Python interpreter (i.e., **not** iPython). Try running it with iPython, which will spit you back into an interactive session once the script is done, and thus keep the plot window open.
* Alternatively, your default ``matplotlib`` settings may have caused this. Check out your ``matplotlibrc`` file (in ``$HOME/.matplotlibrc``) and make sure ``interactive : True``. 

Future versions of *ares* may use blocking commands to ensure that plot windows don't disappear immediately. Email me if you have strong opinions about this.

``IOError: No such file or directory``
--------------------------------------
There are a few different places in the code that will attempt to read-in lookup tables of various sorts. If you get any error that suggests a required input file has not been found, you should:

- Make sure you have set the ``$ARES`` environment variable. See the :doc:`install` page for instructions.
- Make sure the required file is where it should be, i.e., nested under ``$ARES/input``.

In the event that a required file is missing, something has gone wrong. Many lookup tables are downloaded automatically when you run the ``setup.py`` script, so the first thing you should do is re-run ``python setup.py install``. 

``LinAlgError: singular matrix``
--------------------------------
This is an odd one, known to occur in ``ares.physics.Hydrogen`` when using ``scipy.interpolate.interp1d`` to compute the collisional coupling coefficients for spin-exchange. 

We still aren't sure why this happens -- it cannot always be reproduced, even by two users using the same version of *scipy*. A temporary hack is to use linear interpolation, instead of a spline, or to hack off data points at high temperatures in the lookup table. To do this, pass ``interp_cc='linear'`` to whatever routines you're running (it will make its way to the ``Hydrogen`` class). To set it as a default on your system, have a look at the ''Custom Defaults'' section of :doc:`params`.

Working on a more satisfying solution...email me if you encounter this problem.



