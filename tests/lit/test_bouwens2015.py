"""

test_bouwens2015.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri Nov  6 13:33:37 PST 2015

Description: Compare to their Figure 8.

"""

import ares
import numpy as np
import matplotlib.pyplot as pl

b15 = ares.util.read_lit('bouwens2015')

for z in b15.redshifts:
    data = b15.data['lf'][z]
    
    pl.errorbar(data['M'], data['phi'], yerr=data['err'], 
        fmt='o', label=r'$z=%.2g$' % z)

