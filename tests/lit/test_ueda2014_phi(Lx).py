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

u14 = ares.util.read_lit('ueda2014_phi(Lx)')

z = np.linspace(2.4, 3.0, 100)
L = np.logspace(42.5, 47, 100)

fig1 = pl.figure(1); ax1 = fig1.add_subplot(111)

print u14.qsolf_LDDE_hardpars.keys()
results = []
r = []
for i, redshift in enumerate(z):
    model1 = []
    results.append(model1)
    r.append(redshift)
    for p, Lx in enumerate(L):
        model = u14._LuminosityFunction_LDDE(Lx, redshift,\
        u14.qsolf_LDDE_hardpars.values()[12][i],\
        u14.qsolf_LDDE_hardpars.values()[0],\
        u14.qsolf_LDDE_hardpars.values()[3][i],\
        u14.qsolf_LDDE_hardpars.values()[2][i],\
        u14.qsolf_LDDE_hardpars.values()[9][i],\
        u14.qsolf_LDDE_hardpars.values()[6],\
        u14.qsolf_LDDE_hardpars.values()[7],\
        u14.qsolf_LDDE_hardpars.values()[13][i],\
        u14.qsolf_LDDE_hardpars.values()[11][i],\
        u14.qsolf_LDDE_hardpars.values()[8],\
        u14.qsolf_LDDE_hardpars.values()[5][i],\
        u14.qsolf_LDDE_hardpars.values()[10],\
        u14.qsolf_LDDE_hardpars.values()[4][i],\
        u14.qsolf_LDDE_hardpars.values()[1])
        model1.append(model)

          
for i, j in enumerate(results):
    pl.loglog(L, results[i], color = 'r', alpha = 0.25)

pl.title(r'The 2-10 KeV Band at $z$ ~ $%.1f - %.1f$' % (r[0], r[-1]))
pl.ylim(10**-12, 10**-4)
ax1.set_xlabel(r'$L_X$')    
ax1.set_ylabel(r'$\phi(L_X)$')
ax1.legend(loc='best')
pl.show()