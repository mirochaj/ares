"""

test_sed_mcd.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu May  2 10:46:44 2013

Description: Plot a simple multi-color disk accretion spectrum.

"""

import ares
import numpy as np
import matplotlib.pyplot as pl

rmax = 1e2
mass = 10.
f_scatter = [0.1, 0.5]
gamma = [-2.5]
Emin = 10.

simpl = \
{
 'source_type': 'bh', 
 'source_mass': mass,
 'source_rmax': rmax,
 'source_sed': 'simpl',
 'source_Emin': Emin,
 'source_Emax': 5e4,
 'source_EminNorm': 500.,
 'source_EmaxNorm': 8e3,
 'source_alpha': -0.5,
 'source_fsc': 1.0,
 'source_logN': 22.,
}

mcd = \
{
 'source_type': 'bh', 
 'source_sed': 'mcd',
 'source_mass': mass,
 'source_rmax': rmax,
 'source_Emin': Emin,
 'source_Emax': 5e4,
 'source_EminNorm': 500.,
 'source_EmaxNorm': 8e3,
}

bh_mcd = ares.sources.BlackHole(init_tabs=False, **mcd)
bh1 = ares.analysis.Source(bh_mcd)
ax = bh1.PlotSpectrum(color='k')
    
ls = ['-', '--', ':']
colors = ['b', 'g', 'r', 'm']
for i, fsc in enumerate(f_scatter):
    simpl.update({'source_fsc': fsc})
    for j, alpha in enumerate(gamma):
        simpl.update({'source_alpha': alpha})
        
        bh_simpl = ares.sources.BlackHole(init_tabs=False, **simpl)
        bh2 = ares.analysis.Source(bh_simpl)
        
        if j == 0:
            label = r'$f_{\mathrm{sc}} = %g$' % fsc
        else:
            label = None
            
        ax = bh2.PlotSpectrum(color=colors[i], ls=ls[j], label=label)
        pl.draw()
        
ax.legend(loc='lower left')
ax.set_ylim(1e-8, 1e-3)
ax.set_xlim(1e2, 6e4)
ax.fill_betweenx([1e-8, 1e-3], 5e2, 8e3, alpha=0.5, color='gray')
pl.draw()

