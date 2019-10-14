:orphan:

Phenomenological Models for the Global 21-cm Signal
===================================================
Two common phenomenological parameterizations for the global 21-cm signal are included in *ARES* and get their own set of pre-defined parameters: the tanh and Gaussian models. To generate them (without default parameters) one need only do:

:: 

    import ares
    import numpy as np
    import matplotlib.pyplot as pl
    
    sim_1 = ares.simulations.Global21cm(tanh_model=True)
    sim_2 = ares.simulations.Global21cm(gaussian_model=True)
    
    # Have a look
    ax, zax = sim_1.GlobalSignature(color='k', fig=1)
    ax, zax = sim_2.GlobalSignature(color='b', ax=ax)
    
Now, you might say "I could have done that myself extremely easily." You'd be right! However, sometimes there's an advantage in working through *ARES* even when using simply parametric forms for the global 21-cm signal. For example, you can tap into *ARES*' inference module and fit data, perform forecasting, or run large sets of models. In each of these applications, *ARES* can take care of some annoying things for you, like tracking the quantities you care about and saving them to disk in a format that can be easily analyzed later on. For more concrete examples, check out the following pages:
    
    * :doc:`example_inline_analysis`
    * :doc:`example_mcmc_gs`
    * :doc:`example_mc_sampling`
    * :doc:`example_mcmc_analysis`
        
In the remaining sections we'll cover different ways to parameterize the signal.

Parameterizing the IGM 
----------------------
Whereas the Gaussian absorption model makes no link between the brightness temperature and the underlying quantities of interest (ionization history, etc.), the tanh model first models :math:`J_{\alpha}(z)`, :math:`T_K(z)`, and :math:`x_i(z)`, and from those histories produces :math:`\delta T_b(z)`.

Now, let's assemble a set of parameters that will generate a global 21-cm signal using ParameterizedQuantity objects for each main piece: the thermal, ionization, and Ly-:math:`\alpha` histories. We'll assume that the thermal and ionization histories are *tanh* functions, but take the Ly-:math:`\alpha` background evolution to be a power-law in redshift:

::

    pars = \
    {
        'problem_type': 100,           # blank slate global 21-cm signal problem
        'parametric_model': True,      # in lieu of, e.g., tanh_model=True
        
        # Lyman alpha history first: ParameterizedQuantity #0
        'pop_Ja': 'pq[0]',
        'pq_func[0]': 'pl',         # Ja(z) = p0 * ((1 + z) / p1)**p2
        'pq_func_var[0]': '1+z',
        'pq_func_par0[0]': 1e-9,
        'pq_func_par1[0]': 20.,
        'pq_func_par2[0]': -7.,
        
        # Thermal history: ParameterizedQuantity #1
        'pop_Tk': 'pq[1]',         # Tk(z) = p1 + (p0 - p1) * 0.5 * (1 + tanh((p2 - z) / p3))
        'pq_func[1]': 'tanh_abs',
        'pq_func_var[1]': 'z',
        'pq_func_par0[1]': 1e3,
        'pq_func_par1[1]': 0.,
        'pq_func_par2[1]': 8.,
        'pq_func_par3[1]': 6.,
        
        # Ionization history: ParameterizedQuantity #2
        'pop_xi': 'pq[2]',        # xi(z) = p1 + (p0 - p1) * 0.5 * (1 + tanh((p2 - z) / p3))
        'pq_func[2]': 'tanh_abs',
        'pq_func_var[2]': 'z',
        'pq_func_par0[2]': 1,
        'pq_func_par1[2]': 0.,
        'pq_func_par2[2]': 8.,
        'pq_func_par3[2]': 2.,
    }
    
.. note :: The thermal history automatically includes the adiabatic cooling term, so users need not add account for that explicitly.

To run it, as always:

::

    sim_3 = ares.simulations.Global21cm(**pars)
    sim_3.GlobalSignature(color='r', ax=ax)
    pl.savefig('ares_gs_phenom.png')
    
.. figure::  https://www.dropbox.com/s/qo3o3tc7qqk2s5t/ares_gs_phenom.png?raw=1
   :align:   center
   :width:   600

   Comparing three phenomenological models for the global 21-cm signal. 
    

Now, because the parameters of these models are hard to intuit ahead of time, it can be useful to run a set of them. As per usual, we can use some built-in machinery.

::

    blob_pars = ares.util.BlobBundle('gs:basics') \
              + ares.util.BlobBundle('gs:history')

    base_pars = pars.copy()
    base_pars.update(blob_pars)
    
    mg = ares.inference.ModelGrid(**base_pars)
    
Let's focus on the :math:`J_{\alpha}(z)` parameters:

::

    mg.axes = {'pq_func_par1[0]': np.arange(15, 26, 1), 
               'pq_func_par2[0]': np.arange(-9, -2.5, 0.5)}
    
    mg.run('test_Ja_pl', clobber=True)
    
Just to do a quick check, let's look at where the absorption minimum occurs in this model grid:

::

    anl = ares.analysis.ModelSet('test_Ja_pl')
    
    anl.Scatter(anl.parameters, c='z_C', fig=4, edgecolors='none')
    
    pl.savefig('ares_gs_Ja_grid.png')
    
.. figure::  https://www.dropbox.com/s/vvu5gy2wi96s0u0/ares_gs_Ja_grid.png?raw=1
   :align:   center
   :width:   600

   Basic exploration of a 2-D parameter grid.
    


.. Parameterizing Sources
.. ----------------------











.. Sanity Check
.. ------------



  