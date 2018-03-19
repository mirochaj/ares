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

pars = \
{
 'source_temperature': 1e4,
 'source_Emin': 1.,
 'source_Emax': 1e2,
 'source_qdot': 1e50,
}

ls = [':', '--', '-']
for i, logT in enumerate([4, 4.5, 5]):
    pars.update({'source_temperature': 10**logT})

    src = ares.sources.Star(init_tabs=False, **pars)
    bh = ares.analysis.Source(src)
    
    ax = bh.PlotSpectrum(ls=ls[i], 
        label=r'$T_{{\ast}} = 10^{{{:.2g}}} \mathrm{{K}}$'.format(logT))

ax.plot([10.2]*2, [1e-8, 1], color='r', ls='--')
ax.plot([13.6]*2, [1e-8, 1], color='r', ls='--')

ax.legend(loc='lower left')
ax.set_ylim(1e-8, 1)
pl.draw()


