Models with Multiple Source Populations
=========================================
ARES can handle an arbitrary number of source populations. To
access this functionality, create a dictionary representing each source
population of interest. Below, we'll create one population of PopII stars and
one of PopIII stars:

::  

    # a PopII-like model
    src1 = \
    {
     'Tmin': 1e4,               # atomic cooling halos
     'source_type': 'star',
     'fstar': 1e-1,
     'Nion': 4e3,
     'Nlw': 9600.,
    }

    # a PopIII-like model    
    src2 = \
    {
     'Tmin': 300.,              # molecular cooling halos
     'source_type': 'star',
     'fstar': 1e-4,
     'Nion': 30e4,
     'Nlw': 4800.,
    }
    
To run a simulation with each of these populations, we use the ``source_kwargs``
parameter, which must contain a list of dictionaries (one per source population):    

::

    import ares
        
    # Dual-population model
    sim = ares.simulations.Global21cm(source_kwargs=[src1, src2])
    sim.run()
    
    anl = ares.analysis.Global21cm(sim)
    ax = anl.GlobalSignature(color='k', label=r'dual-pop')

For comparison, the same simulation with the PopII-like population only:

::

    sim2 = ares.simulations.Global21cm(**src1)
    sim2.run()
    
    anl2 = ares.analysis.Global21cm(sim2)
    ax = anl2.GlobalSignature(ax=ax, color='b', label='single-pop')
    
Note in the final plot command, we supplied the previous ``ax`` object to overplot
the results of the single population calculation on the same axes as before.

Alternative Technique
----------------------
To avoid use of the ``source_kwargs`` parameter, you can instead add a 
suffix to parameters to denote the population ID number. For example, 
the following is equivalent to the approach taken above:

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
     'Nion{1}': 30e4,
     'Nlw{1}': 4800.,
    }

    import ares
        
    # Dual-population model
    sim = ares.simulations.Global21cm(**pars)
    
    # <run, analyze, etc. just as before>

The integers within curly braces are identification numbers used to keep 
track of the different populations internally.



    