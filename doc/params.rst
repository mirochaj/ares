:orphan:

Parameters
==========
We use keyword arguments to pass parameters around to various *ares* routines. 
A complete listing of parameters and their default values can be found in 
``ares.util.SetDefaultParameterValues.py``. 

Here, we'll provide a brief description of each parameter.

 * :doc:`params_grid`
 * :doc:`params_physics`
 * :doc:`params_sources`
 * :doc:`params_populations`
 * :doc:`params_spectrum`
 * :doc:`params_sfe`
 * :doc:`params_inference`
 * :doc:`params_hmf`
 * :doc:`params_control`
 * :doc:`params_cosmology`
 
Custom Defaults
--------------- 
To adapt the defaults to your liking *without* modifying the source code (all
defaults set in ``ares.util.SetDefaultParameterValues.py``), open the file::

    $HOME/.ares/defaults.py

which by default contains nothing::

    pf = {}
    
To craft your own set of defaults, simply add elements to the ``pf`` dictionary.
For example, if you want to use a default star-formation efficiency of 5% rather
than 10%, open ``$HOME/.ares/defaults.py`` and do::

    pf = {'fstar': 0.05}
    
That's it! Elements of ``pf`` will override the defaults listed in
``ares.util.SetDefaultParameterValues.py`` at run-time.

Alternatively, within a python script you can modify defaults by doing ::

    import ares
    ares.rcParams['fstar'] = 0.05
    
This is similar to how things work in matplotlib (with the ``matplotlibrc`` 
file and ``matplotlib.rcParams`` variable).

Custom Axis-Labels
-------------------
You can do the analogous thing for axis labels (all
defaults set in ``ares.util.Aesthetics.py``). Open the file::

    $HOME/.ares/labels.py

which by default contains nothing::

    pf = {}
    
If you wanted to change the default axis label for the 21-cm brightness
temperature, from :math:`\delta T_b \ (\mathrm{mK})` to :math:`T_b`, you would
do::

    pf = {'dTb': r'$T_b$'}
    
This change will automatically propagate to all built-in analysis routines.



    

  


