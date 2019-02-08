"""

test_sed_agn.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu May  2 10:46:44 2013

Description: Plot Sazonov et al. (2004) AGN template SED.

"""

import ares
import numpy as np
import matplotlib.pyplot as pl

pars = \
{
 'source_type': 'bh', 
 'source_sed': 'sazonov2004',
 'source_Emin': 1,
 'source_Emax': 3e4,
}

src = ares.sources.BlackHole(init_tabs=False, **pars)
bh = ares.analysis.Source(src)

ax = bh.PlotSpectrum()

ax.set_ylim(1e-6, 1e-3)
pl.draw()

