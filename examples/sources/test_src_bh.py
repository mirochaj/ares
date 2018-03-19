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
 'source_Emin': 10.,
 'source_Emax': 1e4,
 'source_sed': 'mcd',
 'source_mass': 10,
}

ls = [':', '--', '-']
for i, logM in enumerate([0, 1, 2]):
    pars.update({'source_mass': 10**logM})

    src = ares.sources.BlackHole(init_tabs=False, **pars)
    bh = ares.analysis.Source(src)
    
    ax = bh.PlotSpectrum(ls=ls[i], 
        label=r'$M_{\bullet} = {} \ M_{\odot}$'.format(int(10**logM)))

ax.plot([10.2]*2, [1e-8, 1], color='r', ls='--')
ax.plot([13.6]*2, [1e-8, 1], color='r', ls='--')

ax.legend(loc='lower left')
ax.set_ylim(1e-8, 1e-3)
pl.draw()


