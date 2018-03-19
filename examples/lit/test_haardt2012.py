"""

test_haardt2012.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sun Jun 21 14:14:45 MDT 2015

Description: 

"""

import ares
import numpy as np
import matplotlib.pyplot as pl

hm12 = ares.util.read_lit('haardt2012')

z, E, flux = hm12.MetaGalacticBackground()

for redshift in [0, 3, 4, 5]:
    i = np.argmin(np.abs(redshift - z))
    pl.loglog(E, flux[i], label=r'$z={:.3g}$'.format(z[i]))

pl.xlim(1e-1, 1e5)
pl.ylim(1e-28, 1e-18)
pl.xlabel(ares.util.labels['E'])
pl.ylabel(ares.util.labels['flux_E'])
pl.legend(loc='upper right', fontsize=14)
