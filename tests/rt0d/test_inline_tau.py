"""

test_inline_tau.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sun Nov  9 15:04:52 MST 2014

Description: 

"""

import os, ares
import matplotlib.pyplot as pl

pars = \
{
 'initial_redshift': 40.,
 'final_redshift': 6.,
 
 'source_type{0}': 'star',
 'source_temperature{0}': 1e4,
 'fstar{0}': 1e-1,
 'Nion{0}': 4e3,
 'Nlw{0}': 9690.,
 'is_heat_src_igm{0}': False,
 'is_ion_src_cgm{0}': True,
 'is_ion_src_igm{0}': False,
 'norm_by{0}': 'lw',
 'approx_lya{0}': True,
 
 'source_type{1}': 'bh',
 'fstar{1}': 1e-1,
 'fX{1}': 10.,
 'norm_by{1}': 'xray',
 'is_lya_src{1}': False,
 'is_ion_src_cgm{1}': False,
 'is_ion_src_igm{1}': True,
 'is_heat_src_igm{1}': True,
 'approx_xray{1}': False,
 'load_tau{1}': True,
 'redshift_bins{1}': 400,
 'spectrum_type{1}': 'pl',
 'spectrum_alpha{1}': -1.5,
 'spectrum_Emin{1}': 2e2,
 'spectrum_Emax{1}': 3e4,
}

ax1 = pl.subplot(311)
ax2 = pl.subplot(312)
ax3 = pl.subplot(313)

ls = '-', '--', ':', '-.'
for i, xswitch in enumerate([1e-3, 0.01, 0.1, 1.0]):
    pars.update({'EoR_xavg': xswitch})

    # Multi-pop model, one with real RT
    sim = ares.simulations.Global21cm(**pars)
    sim.run()
    
    ax1.semilogy(sim.history['z'], sim.history['igm_Tk'], color='k', ls=ls[i])
    ax2.semilogy(sim.history['z'], sim.history['igm_h_2'], color='k', ls=ls[i])
    ax3.plot(sim.history['z'], sim.history['dTb'], color='k', ls=ls[i])
    
    #anl = ares.analysis.Global21cm(sim)

ax1.set_xlim(5, 40)
ax2.set_xlim(5, 40)
ax3.set_xlim(5, 40)
ax1.set_xlabel(r'$z$')
ax1.set_ylabel(r'$T_k$')
ax2.set_ylabel(r'$x_e$')
pl.draw()
