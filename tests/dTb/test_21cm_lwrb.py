"""

test_21cm_cxrb.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Nov 11 13:02:00 MST 2013

Description: 

"""

import os, ares
import matplotlib.pyplot as pl

pars = \
{
 'norm_by': 'lw',
 'source_type': 'star',
 'source_temperature': 1e4,
 
 'approx_lwb': False, 
 'discrete_lwb': True,
 
 'spectrum_type': 'bb',
 'spectrum_Emin': 10.2,
 'spectrum_Emax': 13.6,
 'spectrum_EminNorm': 0.01,
 'spectrum_EmaxNorm': 5e2,
 'lya_nmax': 15, 
 
 'lya_injected': False,
  
}

# Multi-pop model, one with real RT
sim = ares.simulations.Global21cm()
sim.run()

anl = ares.analysis.Global21cm(sim)
ax = anl.GlobalSignature()

fig2 = pl.figure(2)
ax2 = fig2.add_subplot(111)
ax2.semilogy(sim.history['z'], sim.history['Ja'], color='k')

ls = '-', '--'
colors = 'b', 'g', 'm'
for j, injected in enumerate([True, False]):
    tpB = []

    for i, logT in enumerate([4, 4.5, 5]):
        pars.update({'source_temperature': 10**logT, 'lya_injected': injected})
        sim2 = ares.simulations.Global21cm(**pars)
        sim2.run()

        if j == 0:
            label = r'$T_{\ast} = 10^{%.2g} \ \mathrm{K}$' % (logT)
        else:
            label = None

        anl2 = ares.analysis.Global21cm(sim2)
        anl2.GlobalSignature(ax=ax, color=colors[i], label=label, ls=ls[j])

        ax2.semilogy(sim2.history['z'], sim2.history['Ja'], color=colors[i])

        #tpB.append(anl2.turning_points['B'])
    
ax2.set_xlim(5, 50)
pl.show()

for element in tpB:
    print element
