:orphan:

Models with Multiple Source Populations
=========================================
*ares* can handle an arbitrary number of source populations. To
access this functionality, create a dictionary representing each source
population of interest. Below, we'll create one population of PopII stars and
one of PopIII stars:

::  

    pars = \
    {
     'pop_Tmin{0}': 1e4,               # atomic cooling halos
     'pop_type{0}': 'toy', #?
     'pop_fstar{0}': 1e-1,
     'pop_Nion{0}': 4e3,
     'pop_Nlw{0}': 9600.,
     'pop_fX{0}': 1.
     
     'pop_Tmin{1}': 300.,              # molecular cooling halos
     'pop_Tmax{1}': 1e4,             
     'pop_fstar{1}': 1e-4,
     'pop_Nion{1}': 30e3,
     'pop_Nlw{1}': 4800.,
     'pop_fX{0}': 10.,
    }

Now, initialize the simulation class in the usual way:

::

    import ares
        
    # Dual-population model
    sim = ares.simulations.Global21cm(**pars)

The parameter file will be sorted such that each population has its own complete version. For example, to recover the parameter file associated with each population separately:

::

    for key in ['Tmin', 'source_type', 'fstar', 'Nion', 'Nlw']:
        print key, sim.pf.pfs[0][key], sim.pf.pfs[1][key]
    
Just run the thing:

::
    
    sim.run()
    
    ax = sim.GlobalSignature(color='k', label=r'dual-pop')
    

For comparison, the same simulation with the PopII-like population only:

::

    sim2 = ares.simulations.Global21cm(**sim.pf.pfs[0])
    sim2.run()
    
    ax = sim2.GlobalSignature(ax=ax, color='b', label='single-pop')
    
Note in the final plot command, we supplied the previous ``ax`` object to overplot the results of the single population calculation on the same axes as before.

Alternative Syntax
~~~~~~~~~~~~~~~~~~
Using the curly braces to denote population ID numbers will lead to problems if you don't want to create a dictionary of parameters, but instead want to supply the parameters as keyword arguments directly to a simulation class. For this reason, it is also acceptable to bracket population ID numbers with underscores in parameter names. For example, the following is perfectly acceptable, and will lead to the same results as those above.

::

    pars = \
    {
     'Tmin_0_': 1e4,               # atomic cooling halos
     'source_type_0_': 'star',
     'fstar_0_': 1e-1,
     'Nion_0_': 4e3,
     'Nlw_0_': 9600.,
     
     'Tmin_1_': 300.,              # molecular cooling halos
     'source_type_1_': 'star',
     'fstar_1_': 1e-4,
     'Nion_1_': 30e3,
     'Nlw_1_': 4800.,
    }
    
or just

::
    
    sim = ares.simulations.Global21cm(Tmin_0_=1e4, source_type_0_='star',
        fstar_0_=1e-1, Nion_0_=4e3, Nlw_0_=9600., Tmin_1_=300.,
        source_type_1_='star', fstar_1_=1e-4, Nion_1_=3e4,
        Nlw_1_=4800.)


Linking Populations
--------------------
If you are fitting a realization of the 21-cm signal with a multi-population model, you may want to have parameters common to both models that are allowed to vary. To link two parameters together, you can do the following:

::

    pars = \
    {
     'Tmin{0}': 1e4,               # atomic cooling halos
     'source_type{0}': 'star',
     'fstar{0}': 1e-1,
     'Nion{0}': 4e3,
     'Nlw{0}': 9600.,
     
     'Tmin{1}': 300.,              # molecular cooling halos
     'source_type{1}': 'star',
     'fstar{1}': 1e-4,
     'Nion{1}': 'Nion{0}',         # Linked Nion of population #1 to that of population #0
     'Nlw{1}': 4800.,
    }

    import ares
        
    # Dual-population model
    sim = ares.simulations.Global21cm(**pars)
    
    # <run, analyze, etc. just as before>



    