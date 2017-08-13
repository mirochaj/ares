"""

test_atek2015.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri Nov  6 13:33:37 PST 2015

Description: Compare to their Figure 8.

"""

import ares
import numpy as np
import matplotlib.pyplot as pl

# Remember: they are stored as the log10!
atek15 = ares.util.read_lit('atek2015')

for z in atek15.redshifts:
    data = atek15.data['lf'][z]
    
    pl.errorbar(data['M'], np.array(data['phi']), yerr=data['err'], 
        fmt='o', label=r'$z=%.2g$ (Atek)' % z)

pl.xlabel(r'$M_{\mathrm{UV}}$')
pl.ylabel(r'$\log_{10} \phi \ (\mathrm{cMpc}^{-3} \ \mathrm{mag}^{-1})$')
