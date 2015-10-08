"""

test_mpm_w_mgb.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Mar 23 09:35:41 MDT 2015

Description: Test the evolution of a MultiPhaseMedium using ionization and
heating rates induced by a MetaGalacticBackground (hence, mpm_w_mgb). 

"""

import ares
import numpy as np
import matplotlib.pyplot as pl

pars = \
{
 'pop_type': 'galaxy',
 'pop_sed': 'pl',
 'pop_alpha': -1.5,
 'pop_Emin': 2e2,
 'pop_Emax': 3e4,
 'pop_EminNorm': 5e2,
 'pop_EmaxNorm': 8e3,
 'pop_yield': 2.6e39,
 'pop_yield_units': 'erg/s/sfr',

 'pop_solve_rte': True,
 'pop_tau_Nz': 400,
 
 'include_cgm': False,
 'initial_redshift': 40.,
 'final_redshift': 10.,
 'secondary_ionization': 1,

 #'is_ion_src_cgm': False,
 #'include_He': False,
}

ax1 = None; ax2 = None
labels = ['optically thin', 'neutral IGM']
for i, tau_approx in enumerate([True, 'neutral']):
    
    pars['pop_approx_tau'] = tau_approx

    sim = ares.simulations.MultiPhaseMedium(**pars)
    sim.run()
    
    anl = ares.analysis.MultiPhaseMedium(sim)
    ax1 = anl.TemperatureHistory(ax=ax1, label=labels[i])
    
    ax2 = anl.IonizationHistory(zone='igm', element='h', fig=2,
        label=labels[i], ax=ax2)

pl.legend()
pl.show()    
    