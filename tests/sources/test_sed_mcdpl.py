"""

test_sed_mcdpl.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu May  2 10:50:01 2013

Description: Reproduce Figure 1 (kind of) from Kuhlen & Madau (2005). 

"""

import ares
import matplotlib.pyplot as pl

bh_pars = \
{
 'source_type': 'bh',
 'source_mass': 150.,
 'source_rmax': 1e3,
 'spectrum_type': ['mcd', 'pl'],
 'spectrum_fraction': [0.5, 0.5],
 'spectrum_alpha': [None, -1.2],
 'spectrum_Emin': [10.2, 13.6],
 'spectrum_Emax': [1e4, 1e4],
 'spectrum_fcol': [1, None],
 'spectrum_logN': [0., 20.], 
}
          
bh = ares.analysis.Source(ares.sources.RadiationSource(init_tabs=False, 
    **bh_pars))

ax = bh.PlotSpectrum()
ax.set_ylim(1e-6, 2e-3)
pl.draw()
