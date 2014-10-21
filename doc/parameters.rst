Parameters
==========
We use keyword arguments to pass parameters around to various ARES routines. 
A complete listing of parameters and their default values can be found in 
``ares.util.SetDefaultParameterValues.py``. 

Here, we'll provide a brief description of each parameter.

 * :doc:`params_grid`
 * :doc:`params_physics`
 * :doc:`params_sources`
 * :doc:`params_populations`
 * :doc:`params_spectrum`
 * :doc:`params_hmf`
 * :doc:`params_control`
 * :doc:`params_cosmology`
 
To adapt these defaults to your liking *without* modifying ``ares.util.SetDefaultParameterValues.py``,
open the file::

    $HOME/.ares/defaults.py

which by default contains nothing::

    pf = {}
    
To craft your own set of defaults, simply add elements to the ``pf`` dictionary.
For example, if you want to use a default star-formation efficiency of 5% rather
than 10%, open ``$HOME/.ares/defaults.py`` and do::

    pf = {'fstar': 0.05}
    
That's it! Elements of ``pf`` will override the defaults listed in     
``ares.util.SetDefaultParameterValues.py`` at run-time.

    

  


