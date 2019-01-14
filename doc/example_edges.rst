:orphan:

Fitting EDGES-like Signals
==========================
In March of 2018, the EDGES collaboration reported an anamolously-strong absorption signal in the sky-averaged spectrum at 78 MHz (`Bowman et al. (2018) <http://adsabs.harvard.edu/abs/2018Natur.555...67B>`_). This is roughly where one might expect the global 21-cm signal, though its amplitude is 2-3x larger than the most extreme cases in a :math:`\Lambda \text{CDM}` framework. This has led to a variety of exotic explanations, such as milli-charged dark (see, e.g., `Barkana (2018) <http://adsabs.harvard.edu/abs/2018Natur.555...71B>`_, `Fialkov et al. (2018) <http://adsabs.harvard.edu/abs/2018PhRvL.121a1101F>`_, `Berlin et al. (2018) <http://adsabs.harvard.edu/abs/2018PhRvL.121a1102B>`_, `Kovetz et al. (2018) <http://adsabs.harvard.edu/abs/2018PhRvD..98j3529K>`_) matter and excess power in the Rayleigh-Jeans tail of the microwave background (see, e.g., `Feng \& Holder (2018) <http://adsabs.harvard.edu/abs/2018ApJ...858L..17F>`_, `Ewall-Wice et al. (2018) <http://adsabs.harvard.edu/abs/2018ApJ...868...63E>`_, `Fraser et al. (2018) <http://adsabs.harvard.edu/abs/2018PhLB..785..159F>`_, `Pospelov et al. (2018) <http://adsabs.harvard.edu/abs/2018PhRvL.121c1103P>`_). These lists are horribly incomplete, so apologies for that!

In *ares*, we have not included any specific models of dark matter. However, in `Mirocha \& Furlanetto (2019) <http://adsabs.harvard.edu/abs/2019MNRAS.483.1980M>`_ we explored two general possibilities:

- A parametric ``excess cooling'' model, in which the thermal history at very high redshift is allowed to depart from the predictions of standard recombination codes in :math:`\Lambda \text{CDM}` cosmologies.
- An astrophysically-generated radio background, whose strength scales with the star formation of galaxies as :math:`L_R \propto f_R \text{SFR}`. The parameter :math:`f_R` was left as a free parameter.
 
In this section, we will show how to run these models within *ares*.

Excess Cooling Models
---------------------
At high-z the temperature of a mean-density gas parcel evolves between :math:`T(z) \propto (1+z)` and :math:`T(z) \propto (1+z)^2` -- the :math:`(1+z)` dependence a signature that Compton scattering tightly couples the CMB, spin, and kinetic temperatures, and the :math:`(1+z)^2` dependence indicating that Compton scattering has become inefficient, allowing the gas to cool adiabatically. This thermal history can be modeled accurately by noting that the *log*-cooling rate, :math:`d\log T/ d\log t`, of a mean-density gas parcel transitions smoothly between -2/3 at very high-z and -4/3 in a matter-dominated cosmology. So, rather than modeling the thermal history directly, we take

.. math::

    \frac{d\log T}{d\log t} = \frac{\alpha}{3} - \frac{(2+\alpha)}{3} \bigg\{1 + \exp \left[-\left(\frac{z}{z_0}\right)^{\beta} \right] \bigg \} \label{eq:Thist}

and integrate to obtain the thermal history. We have constructed this relation such that :math:`\alpha=-4` reproduces the typical thermal history, and while varying $\alpha$ can change the late-time cooling rate, the cooling rate as :math:`z \rightarrow \infty` tends to :math:`d\log T/ d\log t = -2/3`, as it must to preserve the thermal history during the recombination epoch. See Section 2.3.1 of our paper for more discussion of this model.

To switch to this parameteric cooling model, you can add the following updates to any dictionary of parameters you would usually supply to *ares*:

::

    # New base_kwargs
    cold = \
    {
     'load_ics': 'parametric',
     'approx_thermal_history': 'exp',  # other models are available
     'inits_Tk_p0': 189.5850442,     # z0
     'inits_Tk_p1': 1.26795248,      # Beta
     'inits_Tk_p2': -4,              # alpha
    }

The parameter values listed above are adopted to reproduce the standard :math:`\Lambda \text{CDM}` result, which *ares* draws from CosmoRec. To convince yourself of this, go ahead and compare to the standard scenario:

::
	
    import ares
    import matplotlib.pyplot as pl
    
    # Default simulation, turn off astrophysical sources.
    sim1 = ares.simulations.Global21cm(radiative_transfer=False)
    sim1.run()
    
    ax, zax = sim1.GlobalSignature(label='CosmoRec')
    
    # Use the parameteric cooling
    sim2 = ares.simulations.Global21cm(radiative_transfer=False, **cold)
    sim2.run()
    
    sim2.GlobalSignature(ax=ax, color='b', ls='--', label=r'$\alpha=-4$')
	
Now, if you make cooling faster than adiabatic:

::

    colder = cold.copy()
    colder['inits_Tk_p2'] = -6
    
    sim3 = ares.simulations.Global21cm(radiative_transfer=False, **colder)
    sim3.run()
    
    sim3.GlobalSignature(ax=ax, color='r', label=r'$\alpha=-6$')
    ax.legend()
    pl.savefig('ares_edges_cold.png')
	
*Voila!*

.. figure::  https://www.dropbox.com/s/e9ukxjkqndeqkvj/ares_edges_cold.png?raw=1
   :align:   center
   :width:   600

   Comparison of parametric excess cooling models with CosmoRec solution for :math:`\Lambda \text{CDM}` cosmology.

By the way, if you would like to add the EDGES models you can do so via

::

    b18 = ares.util.read_lit('bowman2018')
    b18.plot_recovered(ax=ax, color='k', alpha=0.2)


If you want to use the exact models presented in `Mirocha \& Furlanetto (2019) <http://adsabs.harvard.edu/abs/2019MNRAS.483.1980M>`_, you can summon the requisite parameters using the ``ParameterBundle`` framework.

::

    pars = ares.util.ParameterBundle('mirocha2019:base') \
         + ares.util.ParameterBundle('mirocha2019:cold')
         
    sim4 = ares.simulations.Global21cm(**pars)    
    sim4.run()
    
    sim4.GlobalSignature(ax=ax, color='g', lw=3, ls='--', label='MF18 cooling', ymin=-600)
    ax.legend()
    
    pl.savefig('ares_edges_mf18_cooling.png')
    
.. figure::  https://www.dropbox.com/s/vfq7te1xqn39w1o/ares_edges_mf18_cooling.png?raw=1
   :align:   center
   :width:   600

   Various ``excess cooling'' models for the global 21-cm signal compared to the EDGES 78 MHz signal(s).
    

	
Astrophysical Radio Backgrounds
-------------------------------
The simplest way to augment the radio background is to parameterize it. You can do so easily in *ares* via the parameter ``Tbg``, to which you can supply a Python function (assumed to be defined in terms of redshift), or ``pl``, to indicate use of a power-law model. In the latter case, you must also supply the parameters ``Tbg_p0``, ``Tbg_p1``, and ``Tbg_p2`` which define the power-law as

.. math::
    T_r(z) = p_0 \left(\frac{1+z}{1+p_1} \right)^{p_2}

Another way to implement a new radio background is to link emission to star formation, in analogy with how we generally scale the cosmic UV and X-ray backgrounds. In `Mirocha \& Furlanetto (2019) <http://adsabs.harvard.edu/abs/2019MNRAS.483.1980M>`_, we adopted an empirical relation between the monochromatic 1.4 GHz luminosity and SFR (see, e.g., `Gurkan et al. 2018 <http://adsabs.harvard.edu/abs/2018MNRAS.475.3010G>`_),

.. math::
    L_R = 10^{22} f_R \left(\frac{\text{SFR}}{M_{\odot} \ \mathrm{yr}^{-1}} \right) \ \text{W} \ \text{s}^{-1} \ \text{Hz}^{-1}

and assumed a power-law spectrum with index :math:`\alpha=-0.7`.

.. note:: We have scaled the Gurkan et al. 150 MHz normalization to 1.4 GHz. See Section 2.3.2 in our paper for more details.

To create such a source population in *ares*, we build off the standard approach calibrated to high-z UV luminosity functions. Because we're assuming that the radio spectrum does not depend on host galaxy mass or time, we can simply link the SFRD of this new population to that of a pre-existing population, in this case with ID number 0:

::

    from ares.physics.Constants import nu_0_mhz, h_p, erg_per_ev
        
    # Need rest 21-cm frequency and spectral bounds in eV
    E21 = nu_0_mhz * 1e6 * h_p / erg_per_ev # 1.4 GHz -> eV
    Emin = 1e7 * (h_p / erg_per_ev)         # 10 MHz -> eV
    Emax = 1e12 * (h_p / erg_per_ev)        # 100 GHz -> eV
    
    # Setup new parameters
    radio_pop = \
    {
     'pop_sfr_model{2}': 'link:sfrd:0',    # Link to SFRD of population #0
     'pop_sed{2}': 'pl',                 
     'pop_alpha{2}': -0.7,
     'pop_Emin{2}': Emin,                  
     'pop_Emax{2}': Emax,
     'pop_EminNorm{2}': None,
     'pop_EmaxNorm{2}': None,
     'pop_Enorm{2}': E21, # 1.4 GHz
     'pop_rad_yield_units{2}': 'erg/s/sfr/hz', # Indicate this normalization is monochromatic
     
     'pop_solve_rte{2}': True,
     'pop_radio_src{2}': True,  # Only emit in the radio
     'pop_lw_src{2}': False,
     'pop_lya_src{2}': False,
     'pop_heat_src_igm{2}': False,
     'pop_ion_src_igm{2}': False,
     'pop_ion_src_cgm{2}': False,
     
     # Key parameters!
     'pop_rad_yield{2}': 1e22,
     'pop_zdead{2}': None,
    }
    
    
.. note:: If you're running with a different set of baseline parameters, you 
    may need to use a different population ID number!
    
Again, these parameters are stored as a ``ParameterBundle``, with the best-fitting values used in our paper, so we can simply execute:
    
:: 
   
    pars = ares.util.ParameterBundle('mirocha2019:base') \
         + ares.util.ParameterBundle('mirocha2019:radio')    
         
    sim5 = ares.simulations.Global21cm(**pars)  
    sim5.run()
    
    sim5.GlobalSignature(ax=ax, color='r', label='MF18 radio', ymin=-600)
    ax.legend()
    
    pl.savefig('ares_edges_mf18_radio.png')
    
.. figure::  https://www.dropbox.com/s/07u8a9m3i01ctve/ares_edges_mf18_radio.png?raw=1
   :align:   center
   :width:   600

   Same figure as before, with the addition of a source-generated radio background.
    
    
    
By default, the parameter ``pop_zdead{2}`` is used, which instantaneously shuts down the radio emission at the supplied redshift. To see the impact of this, simply set ``pop_zdead{2}`` to ``None``.
    
    
    