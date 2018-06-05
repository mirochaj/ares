:orphan:

Models with Multiple Source Populations
=========================================
*ares* can handle an arbitrary number of source populations. To
access this functionality, create a dictionary representing each source
population of interest. Below, we'll create a population representative of PopII stars and another representative of PopIII stars.

Before we start, it is important to note that in *ares*, source populations are identified by their spectra over some contiguous interval in photon energy. This can be somewhat counterintuitive. For example, though UV emission from stars and X-ray emission from their compact remnants, e.g., X-ray binary systems, are both natural byproducts of star formation, we treat them as separate source populations in *ares* even though the emission from each type of source is related to the same rate of star formation. However, because stars and XRBs have very different spectra, whose normalizations are parameterized differently, it is more convenient in the code to keep them separate. Because of this, what you might think of as a single source population (stars and their remnants) actually constitutes *two* source populations in *ares*. 

Let's start with a PopII source population:

::  

    pars = \
    {
     'problem_type': 100,              # Blank slate global 21-cm signal calculation

     # Setup star formation
     'pop_Tmin{0}': 1e4,               # atomic cooling halos
     'pop_fstar{0}': 1e-1,             # 10% star formation efficiency
     
     # Setup UV emission
     'pop_sed_model{0}': True,
     'pop_sed{0}': 'bb',               # PopII stars -> 10^4 K blackbodies
     'pop_temperature{0}': 1e4,
     'pop_rad_yield{0}': 1e42,
     'pop_fesc{0}': 0.2,
     'pop_Emin{0}': 10.19, 
     'pop_Emax{0}': 24.6,
     'pop_EminNorm{0}': 13.6,
     'pop_EmaxNorm{0}': 24.6,
     'pop_lya_src{0}': True,
     'pop_ion_src_cgm{0}': True,
     'pop_heat_src_igm{0}': False,
     
     # Setup X-ray emission
     'pop_sed{1}': 'pl',
     'pop_alpha{1}': -1.5, 
     'pop_rad_yield{1}': 2.6e38,
     'pop_Emin{1}': 2e2, 
     'pop_Emax{1}': 3e4,
     'pop_EminNorm{1}': 5e2,
     'pop_EmaxNorm{1}': 8e3,
     
     'pop_lya_src{1}': False,
     'pop_ion_src_cgm{1}': False,
     'pop_heat_src_igm{1}': True,
     
     'pop_sfr_model{1}': 'link:sfrd:0',
    }
    
.. note :: See :doc:`problem_types` for more information about why we chose ``problem_type=100`` here.    
    
We might as well go ahead and run this to establish a baseline:

::

    import ares

    sim = ares.simulations.Global21cm(**pars)
    sim.run()
    
    ax, zax = sim.GlobalSignature(color='k')
    
Now, let's add a PopIII-like source population. We'll assume that PopIII sources are brighter on average (in both the UV and X-ray) but live in lower mass halos. We could just copy-pase the dictionary above, change the population ID numbers and, for example, the UV and X-ray ``pop_rad_yield`` parameters. Or, we could use some built-in tricks to speed this up.

First, let's take the PopII parameter set and make a ParameterBundle object:

::

    popII = ares.util.ParameterBundle(**pars)
    
This let's us easily extract parameters according to their ID number, i.e.,

::

    popIII_uv = popII.pars_by_pop(0, True)
    popIII_uv.num = 2
    popIII_xr = popII.pars_by_pop(1, True)
    popIII_xr.num = 3

The second argument tells *ares* to remove the parameter ID numbers.

Now, we can simply reset the ID numbers and update a few important parameters:

::

    
    popIII_uv['pop_Tmin{2}'] = 300
    popIII_uv['pop_Tmax{2}'] = 1e4
    popIII_uv['pop_rad_yield{2}'] = 1e43
    popIII_uv['pop_temperature{2}'] = 1e5
    popIII_uv['pop_fstar{2}'] = 1e-3
    
    popIII_xr['pop_sfr_model{3}'] = 'link:sfrd:2'
    popIII_xr['pop_rad_yield{3}'] = 2.6e39
    
Now, let's make the final parameter dictionary and run it:    

::

    pars.update(popIII_uv)
    pars.update(popIII_xr)
    
    sim = ares.simulations.Global21cm(**pars)
    sim.run()
    
    ax, zax = sim.GlobalSignature(color='b', ax=ax)

    import matplotlib.pyplot as pl
    pl.savefig('ares_gs_multipop.png')
    
.. figure::  https://www.dropbox.com/s/otpmvoz8ca7wett/ares_gs_multipop.png?raw=1
   :align:   center
   :width:   600

   Example calculations with a single population (black) and multiple source 
   populations (blue).
    

Note that the parameter file hangs onto the parameters of each population separately. To verify a few key changes, you could do:    

::

    for key in ['pop_Tmin', 'pop_fstar', 'pop_rad_yield']:
        print key, sim.pf.pfs[0][key], sim.pf.pfs[2][key]


    
.. note :: These are very simple models for PopII and PopIII stars. For more 
    sophisticated approaches, see :doc:`example_pop_galaxy` and
    :doc:`example_popIII`.    


Note in the final plot command, we supplied the previous ``ax`` object to overplot the results of the single population calculation on the same axes as before.

Alternative Population ID Tagging Syntax
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Using the curly braces to denote population ID numbers will lead to problems if you don't want to create a dictionary of parameters, but instead want to supply the parameters as keyword arguments directly to a simulation class. For this reason, it is also acceptable to bracket population ID numbers with underscores in parameter names. For example, instead of

::

    pars['pop_Tmin{0}'] = 1e4
    
you could do

::

    pars['pop_Tmin_0_'] = 1e4
    


Linking Populations
~~~~~~~~~~~~~~~~~~~~
If you are fitting a realization of the 21-cm signal with a multi-population model, you may want to have parameters common to both models that are allowed to vary. To link two parameters together, you can simply replace a parameter value of one population (usually a number) to the *name* of a parameter for another population. For example, to make the PopII and PopIII star formation efficiencies the same (using the parameter dictionary above), you could do

::

    pars['pop_fstar{2}'] = 'pop_fstar{0}'
    
and any change to ``pop_fstar{0}`` will automatically propagate to ``pop_fstar{2}``.


    



    