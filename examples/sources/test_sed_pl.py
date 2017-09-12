"""

test_sed_mcd.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu May  2 10:46:44 2013

Description: Plot a simple power-law spectrum.

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
}

ls = [':', '--', '-']
for i, alpha in enumerate([-0.5, -1.5, -2.5]):
    pars.update({'source_alpha': alpha})

    src = ares.sources.BlackHole(init_tabs=False, **pars)
    bh = ares.analysis.Source(src)
    
    ax = bh.PlotSpectrum(ls=ls[i], 
        label=r'$\alpha = {:.2g}$'.format(alpha))

ax.legend(loc='upper right')
ax.set_ylim(1e-8, 1)
pl.draw()


