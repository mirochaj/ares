"""
test_Lx(z)

Author: Jacob Jost
Affiliation: University of Colorado at Boulder (Undergraduate)
Created on: Thu June 4 09:00:00 MDT 2015

This is the plot of the integrated Lx plotted against SFRD

--- = spacing between different sections of code
"""

import ares
import numpy as np
import matplotlib.pyplot as pl
from scipy import integrate

u14 = ares.util.read_lit('ueda2014') 
r15 = ares.util.read_lit('robertson2015')

#--------------------------------------------------------------

z = np.linspace(0.0, 5.0, 100)

integrand = []
r = []
for i in range(len(z)):
    integrand1 = []
    x = lambda Lx: u14.LuminosityFunction_LDDE(Lx, z[i])
    p, err = integrate.quad(x, 10**41, 10**46)
    integrand.append(p)
print integrand

pl.semilogy(z, integrand)

#--------------------------------------------------------------

pop = ares.analysis.Population(r15)
models = pop.SamplePosterior(z, r15._SFRD, r15.sfrd_pars, r15.sfrd_err)

ax = pl.subplot(111)
for i in range(int(models.shape[1])):
    ax.semilogy(z, models[:,i], color='b', alpha=0.05)

# Plot best-fit
ax.semilogy(z, r15.SFRD(z))    
ax.set_xlabel(r'$z$')
ax.set_ylabel(ares.util.labels['sfrd'])
pl.title(r'$SFRD$ $&$ $Lx$')
#--------------------------------------------------------------

pl.show()