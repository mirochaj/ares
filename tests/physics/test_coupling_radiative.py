"""

test_physics_lya_coupling.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Sep 25 10:01:27 MDT 2013

Description: 

"""

import rt1d
import numpy as np
import matplotlib.pyplot as pl

#
##
### 
zarr = np.linspace(20, 40)
###
##
#

hydr1 = rt1d.physics.Hydrogen(approx_Salpha=1)
hydr2 = rt1d.physics.Hydrogen(approx_Salpha=2)
hydr3 = rt1d.physics.Hydrogen(approx_Salpha=3)

pl.plot(zarr, map(lambda z: hydr1.Sa(z, hydr1.cosm.Tgas(z)), zarr), 
    color='k', ls='-')
pl.plot(zarr, map(lambda z: hydr2.Sa(z, hydr2.cosm.Tgas(z)), zarr), 
    color='k', ls='--', label='Chuzhoy, Alvarez, & Shapiro \'05')
pl.plot(zarr, map(lambda z: hydr3.Sa(z, hydr3.cosm.Tgas(z)), zarr), 
    color='k', ls=':', label='Furlanetto & Pritchard \'06')   
     
pl.ylim(0, 1.1)
pl.xlabel(r'$z$')
pl.ylabel(r'$S_{\alpha}$')
pl.legend(loc='lower right', frameon=False)
pl.annotate(r'assuming $T_k \propto (1+z)^2$', (0.1, 0.4), xycoords='axes fraction')