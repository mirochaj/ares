"""

test_inline_tau.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sun Nov  9 15:04:52 MST 2014

Description: 

"""

import os, ares, time
import matplotlib.pyplot as pl

pars = \
{
 'include_He': True,
 'approx_He': False,
 'track_extrema': False,

 'initial_redshift': 50.,
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

mp = ares.analysis.MultiPanel(dims=(3,1))

sims = []
t = []

ls = '-', '--', ':', '-.'
for i, xswitch in enumerate([1e-3, 0.01, 0.1, 1.0]):
    pars.update({'EoR_xavg': xswitch})

    # Multi-pop model, one with real RT
    t1 = time.time()
    sim = ares.simulations.Global21cm(**pars)
    sim.run()
    t2 = time.time()
    
    t.append(t2-t1)
    
    mp.grid[0].semilogy(sim.history['z'], sim.history['igm_Tk'], color='k', 
        ls=ls[i])
    mp.grid[1].semilogy(sim.history['z'], sim.history['igm_h_2'], color='k', 
        ls=ls[i])
    mp.grid[2].plot(sim.history['z'], sim.history['dTb'], color='k', 
        ls=ls[i])
        
    sims.append(sim)
    
print np.array(t) / t[-1]
    
mp.grid[0].set_xlim(5, 40)
mp.grid[1].set_xlim(5, 40)
mp.grid[2].set_xlim(5, 40)
mp.grid[0].set_xlabel(r'$z$')
mp.grid[0].set_ylabel(r'$T_k$')
mp.grid[1].set_ylabel(r'$x_e$')
mp.grid[2].set_ylabel(r'$\delta T_b$')
mp.fix_ticks()
pl.draw()
