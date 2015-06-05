"""
test_Lx(z)

Author: Jacob Jost
Affiliation: University of Colorado at Boulder (Undergraduate)
Created on: Thu June 4 09:00:00 MDT 2015

NOTE: plots may differ than Ueda et al. (2014) because their plots show the De-absorbed
XLF and the redshift ranges arebinned, also my be cutting the plots off at a certain value***
"""

import ares
import numpy as np
import matplotlib.pyplot as pl

u14 = ares.util.read_lit('ueda2014')

z = np.linspace(2.4, 3.0, 100)
L = np.logspace(42.5, 47, 100)

fig1 = pl.figure(1); ax1 = fig1.add_subplot(111)

colors = 'k', 'b', 'g', 'o' 'r', 'p'
models = []
r = []
for i, redshift in enumerate(z):
    model1 = []
    models.append(model1)
    r.append(redshift)
    for p, Lx in enumerate(L):
        model = u14.LuminosityFunction_LDDE(Lx, redshift)
        model1.append(model)
models = np.array(models) 
r = np.array(r)

for i, j in enumerate(models):
    pl.loglog(L, models[i], alpha = 0.25)
    pl.title(r'$The$ $2-10$ $KeV$ $at$ $z$ ~ $%.1f - %.1f$' % (r[0], r[-1]))
    pl.ylim(10**-12, 10**-2)
    pl.xlim(L[0], L[-23])

ax1.set_xlabel(r'$L_X$')    
ax1.set_ylabel(r'$\phi(L_X)$')
ax1.legend(loc='best')
pl.show()