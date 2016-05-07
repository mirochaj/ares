"""

test_inline_tau.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sun Nov  9 15:04:52 MST 2014

Description: 

"""

import ares
import numpy as np
import matplotlib.pyplot as pl

pars = \
{
 'include_He': True,
 'approx_He': False,
 'track_extrema': True,

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
 'approx_lwb{0}': True,
 
 'source_type{1}': 'bh',
 'fstar{1}': 1e-1,
 'fX{1}': 1.0,
 'norm_by{1}': 'xray',
 'is_lya_src{1}': False,
 'is_ion_src_cgm{1}': False,
 'is_ion_src_igm{1}': True,
 'is_heat_src_igm{1}': True,
 'approx_xrb{1}': False,
 'load_tau{1}': True,
 'redshift_bins{1}': 1e3,
 'spectrum_type{1}': 'pl',
 'spectrum_alpha{1}': -1.5,
 'spectrum_Emin{1}': 2e2,
 'spectrum_Emax{1}': 3e4,
}

mp = ares.analysis.MultiPanel(dims=(3,1))

sims = []
t = []

xEoR = [1e-4, 1e-3, 0.01, 0.1, 1.0]

ls = '-.', '-', '--', ':', '-.'
for i, xswitch in enumerate(xEoR):
    pars.update({'EoR_xavg': xswitch})

    sim = ares.simulations.Global21cm(**pars)
    sim.run()    
    
    mp.grid[0].semilogy(sim.history['z'], sim.history['igm_Tk'], color='k', 
        ls=ls[i], label=r'$x_{\mathrm{EoR}} = %.2e$' % xswitch)
    mp.grid[1].semilogy(sim.history['z'], sim.history['igm_h_2'], color='k', 
        ls=ls[i])
    mp.grid[2].plot(sim.history['z'], sim.history['dTb'], color='k', 
        ls=ls[i])
        
    sims.append(sim)
        
mp.grid[0].legend(loc='upper right', fontsize=14, ncol=2)    
    
mp.grid[0].set_xlim(5, 40)
mp.grid[1].set_xlim(5, 40)
mp.grid[2].set_xlim(5, 40)
mp.grid[0].set_xlabel(r'$z$')
mp.grid[0].set_ylabel(r'$T_k$')
mp.grid[1].set_ylabel(r'$x_e$')
mp.grid[2].set_ylabel(r'$\delta T_b$')
mp.fix_ticks()
pl.draw()

# Print header
print 'xEoR         ',
things = ['z', 'T', 'curv']
for tp in ['B', 'C', 'D', 'trans']:
    for j, element in enumerate(sim.turning_points[tp]):
        print '%s_%s           ' % (things[j], tp),
print ''
print '-'* 200    

for i, sim in enumerate(sims):

    s = "%.2e " % xEoR[i]
    for tp in ['B', 'C', 'D', 'trans']:
        for element in sim.turning_points[tp]:
            s += str(element)
            s += ' '
            
    print s        
    
