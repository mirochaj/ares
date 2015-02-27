"""

test_modelgrid.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sun Jun 15 12:59:04 MDT 2014

Description: 

"""

import ares, time
import numpy as np
import matplotlib.pyplot as pl

#
##
fstar = 10**np.arange(-1.0, -0.5, 0.1)
fX = 10**np.arange(-1., 1.5, 0.1)
grid_axes = {'fstar': fstar, 'fX': fX}
##
#

mg = ares.inference.ModelGrid(auto_generate_blobs=True)

mg.set_axes(**grid_axes)
mg.LoadBalance(0)

t1 = time.time()
mg.run('test_grid', clobber=True)
t2 = time.time()

print "Run complete in %.4g minutes.\n" % ((t2 - t1) / 60.)





