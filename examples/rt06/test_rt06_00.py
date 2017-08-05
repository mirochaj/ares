"""

test_rt06_00.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sun Dec 30 22:10:25 2012

Description: Test rate coefficients.

"""

import ares
import numpy as np
import matplotlib.pyplot as pl
import chianti.core as cc

pl.rcParams['legend.fontsize'] = 14

#
dims = 32
T = np.logspace(2, 7, dims)
#

# Chianti rates
h1 = cc.ion('h_1', T)
h2 = cc.ion('h_2', T)
he1 = cc.ion('he_1', T)
he2 = cc.ion('he_2', T)
he3 = cc.ion('he_3', T)

# Analytic fits
coeff = ares.physics.RateCoefficients()

"""
Plot collisional ionization rate coefficients.
"""

h1.diRate()
he1.diRate()
he2.diRate()

fig1 = pl.figure(1)
ax1 = fig1.add_subplot(111)

# Chianti
ax1.loglog(T, h1.DiRate['rate'], color = 'k', ls = '-', 
    label='chianti')
ax1.loglog(T, he1.DiRate['rate'], color = 'k', ls = '--')
ax1.loglog(T, he2.DiRate['rate'], color = 'k', ls = ':')

# Fukugita '94
ax1.loglog(T, coeff.CollisionalIonizationRate(0, T), color = 'b', ls = '-', 
    label = 'Fukugita & Kawasaki \'94')
ax1.loglog(T, coeff.CollisionalIonizationRate(1, T), color = 'b', ls = '--')
ax1.loglog(T, coeff.CollisionalIonizationRate(2, T), color = 'b', ls = ':')

ax1.set_xlim(min(T), max(T))
ax1.set_ylim(1e-14, 1e-7)
ax1.set_xlabel(r'$T \ (\mathrm{K})$')
ax1.set_ylabel(r'Collisional Ionization Rate $(\mathrm{cm}^3 \ \mathrm{s}^{-1})$')
ax1.legend(loc='upper left', frameon=False)

"""
Plot recombination rate coefficients.
"""

h2.rrRate()
he2.rrRate()
he2.drRate()
he3.rrRate()

fig2 = pl.figure(2)
ax2 = fig2.add_subplot(111)


# Chianti
ax2.loglog(T, h2.RrRate['rate'], color = 'k', ls = '-', label = 'chianti')
ax2.loglog(T, he2.RrRate['rate'], color = 'k', ls = '--')
ax2.loglog(T, he3.RrRate['rate'], color = 'k', ls = ':')
ax2.loglog(T, he2.DrRate['rate'], color = 'k', ls = '-.')

# Fukugita
ax2.loglog(T, coeff.RadiativeRecombinationRate(0, T), color = 'b', ls = '-',
    label = 'Fukugita & Kawasaki \'94')
ax2.loglog(T, coeff.RadiativeRecombinationRate(1, T), color = 'b', ls = '--')
ax2.loglog(T, coeff.RadiativeRecombinationRate(2, T), color = 'b', ls = ':')
ax2.loglog(T, coeff.DielectricRecombinationRate(T), color = 'b', ls = '-.')

ax2.set_xlim(min(T), max(T))
ax2.set_ylim(1e-16, 1e-10)
ax2.set_xlabel(r'$T \ (\mathrm{K})$')
ax2.set_ylabel(r'Recombination Rate $(\mathrm{cm}^3 \ \mathrm{s}^{-1})$')

ax2.legend(loc='lower left', frameon=False)

pl.draw()
