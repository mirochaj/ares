"""

test_sed_apl.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu May  2 10:46:44 2013

Description: Plot an absorbed power-law spectrum.

"""

import ares
import numpy as np
import matplotlib.pyplot as pl

pars = \
{
 'source_type': 'bh', 
 'source_sed': 'pl',
 'source_Emin': 2e2,
 'source_Emax': 3e4,
 'source_EminNorm': 5e2,
 'source_EmaxNorm': 8e3, 
}

colors = 'k', 'b', 'g'
ls = ['-.', ':', '--', '-', '-.']
for i, alpha in enumerate([-0.5, -1.5, -2.5]):
    for j, logN in enumerate([-np.inf, 20, 21, 22, 23]):
        pars.update({'source_alpha': alpha, 'source_logN': logN})
        
        src = ares.sources.BlackHole(init_tabs=False, **pars)
        bh = ares.analysis.Source(src)
        
        if j == 2:
            label = r'$\alpha = {:.2g}$'.format(alpha)
        else:
            label = None
        
        ax = bh.PlotSpectrum(color=colors[i], ls=ls[j], label=label)

ax.fill_between([5e2, 8e3], 1e-8, 1, alpha=0.3, color='r')
ax.legend(loc='upper right', fontsize=14)
ax.set_ylim(1e-8, 1)
pl.draw()


