Quick-start
===========
To get a sense for how the code works, let's start just after cosmological recombination, 
and simulate the dark ages 21-cm signal (i.e., signal in the absence of astrophysical sources).

::

    import glorb
    
    params = \
    {
     'radiative_transfer': False,  # No light sources
     'initial_redshift': 800 ,     # Start just after recombination
     'final_redshift': 10,
     'initial_timestep': 1e-8,     # Default is 1e-2 Myr: too big for such high redshifts
    }
    
    sim = glorb.run.Simulation(**params)
    sim.run()
    
    anl = glorb.analysis.Synthetic21cm(sim)
    ax1 = anl.TemperatureHistory(fig=1, xscale='log')
    ax2 = anl.GlobalSignature(fig=2, xscale='log')
    
The results should be similar to those presented in Figure 6 of `Furlanetto, Oh, & Briggs (2006) <http://adsabs.harvard.edu/abs/2006PhR...433..181F>`_.

Now, let's add some astrophysics. This means we turn on ``radiative_transfer``, and
re-run the calculation from :math:`z=40`, which is  when sources first turn on by default
(i.e., ``first_light_redshift=40``).

::

    params['radiative_transfer'] = True
    params['initial_redshift'] = 40
    params['final_redshift'] = 5     
    
    sim2 = glorb.run.Simulation(**params)
    sim2.run()
    
    anl = glorb.analysis.Synthetic21cm(sim2)
    ax1 = anl.TemperatureHistory(ax=ax1, color='b')
    ax2 = anl.GlobalSignature(ax=ax2, color='b')
    
Now, we can see the three astrophysical "turning points" that persist over large
ranges of parameter space. Common astrophysical parameters include the star formation
efficiency, :math:`f_{\ast}` (``fstar``), the normalization of the :math:`L_X`-SFR relation, 
:math:`f_X` (``fX``), and the minimum virial temperature of star-forming halos, :math:`T_{\min}` (``Tmin``),
for example.

To do a simple parameter study, we can pass these parameters to the Simulation class
as keyword arguments:

::

    ax = None; colors = ['k', 'b', 'g']
    for i, fX in enumerate([0.1, 1, 10.]):
        sim = glorb.run.Simulation(final_redshift=5, fX=fX)
        sim.run()
    
        anl = glorb.analysis.Synthetic21cm(sim)
        ax = anl.GlobalSignature(ax=ax, color=colors[i])

There are also built-in routines for doing parameter studies in arbitrary 
N-D spaces. For more info, see :doc:`example_grid_I`.








