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

    