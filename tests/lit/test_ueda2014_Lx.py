"""
test_Lx(z)

Author: Jacob Jost
Affiliation: University of Colorado at Boulder (Undergraduate)
Created on: Thu June 4 09:00:00 MDT 2015

NOTE: 
    plots may differ than Ueda et al. (2014) because their plots show the De-absorbed
    XLF and the redshift ranges arebinned, also my be cutting the plots off at a certain value***

    HEADS UP: this takes a while to run, use caution. 
"""

import ares
import numpy as np
import matplotlib.pyplot as pl
from scipy import integrate

u14 = ares.util.read_lit('ueda2014_phi(Lx)')

z = np.linspace(0, 5, 100)
models = 50

fig1 = pl.figure(1); ax1 = fig1.add_subplot(111)

integrand = []
for i in range(models):
    integrand1 = []
    r = []
    integrand.append(integrand1)
    for j in range(len(z)):
        x = lambda Lx: u14._LuminosityFunction_LDDE(Lx, z[j],\
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
        p, err = integrate.quad(x, 10**41, 10**46)
        r.append(z[j])
        integrand1.append(p)
          
for i, j in enumerate(integrand):
    pl.semilogy(z, integrand[i], color = 'k', alpha = 0.25)

pl.title(r'The 2-10 KeV Band at $z$ ~ $%.1f - %.1f$' % (r[0], r[-1]))

ax1.set_xlabel(r'$z$')    
ax1.set_ylabel(r'$\phi(L_X)$')
ax1.legend(loc='best')
pl.show()