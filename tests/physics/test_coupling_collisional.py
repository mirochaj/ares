"""

test_physics_collisional_coupling.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri Nov  9 16:01:47 2012

Description: 

"""


import numpy as np   
import matplotlib.pyplot as pl
from ares.physics import Hydrogen

hydr = Hydrogen()

# Relevant temperature range
T = np.logspace(0, 4, 500)

fig1 = pl.figure(1)#; fig2 = pl.figure(2)
ax1 = fig1.add_subplot(111)#; ax2 = fig2.add_subplot(111)

# Plot coupling coefficients vs. temperature (tabulated)
ax1.scatter(hydr.tabulated_coeff['T_H'], 
    hydr.tabulated_coeff['kappa_H'], color = 'k')
ax1.scatter(hydr.tabulated_coeff['T_e'], 
    hydr.tabulated_coeff['kappa_e'], color = 'k')
    
# Interpolated values
ax1.loglog(T, hydr.kappa_H(T), color = 'k', ls = '-', 
    label=r'$\kappa_{10}^{\mathrm{HH}}$')
ax1.loglog(T, hydr.kappa_e(T), color = 'k', ls = '--', 
    label=r'$\kappa_{10}^{\mathrm{eH}}$') 
ax1.set_xlim(1, 1e4)
ax1.set_ylim(1e-13, 2e-8)
ax1.set_xlabel(r'$T \ (\mathrm{K})$')
ax1.set_ylabel(r'$\kappa \ (\mathrm{cm}^3 \mathrm{s}^{-1})$')
ax1.legend(loc = 'lower right', frameon = False)
ax1.annotate('Points from Zygelman (2005)', (1.5, 1e-8), ha='left')
ax1.annotate('Lines are spline fit to points', (1.5, 5e-9), ha='left')

pl.show()

# Derivatives!
#Tk = np.logspace(1, 3, 500)
#
#ax2.semilogx(Tk, map(hydr.dlogkH_dlogT, Tk), color='k', ls='-',
#    label=r'$\kappa_{10}^{\mathrm{HH}}$')
#ax2.semilogx(Tk, map(hydr.dlogke_dlogT, Tk), color='k', ls='--',
#    label=r'$\kappa_{10}^{\mathrm{eH}}$')
#
#ax2.set_xlabel(r'$T \ (\mathrm{K})$')
#ax2.set_ylabel(r'$\frac{d\log \kappa}{d\log T_K}$')
#ax2.legend(loc='upper right', frameon=False)
#   
#pl.show()   
  