"""

test_s99.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sat Jun  6 11:12:51 MDT 2015

Description: 

"""

import ares
import matplotlib.pyplot as pl

s99 = ares.util.read_lit('leitherer1999')

# Currently, the starburst99 module works a little differently than others. 
# Since there are so many "switches", rather than "knobs", it gets its own 
# StellarPopulation class to deal with this. Need to work on how to generalize 
# this with the rest of the API.
pop = s99.StellarPopulation(pop_ssp=True, pop_Z=0.04)

# Stellar SED at 1 Myr
fig1 = pl.figure(1); ax1 = fig1.add_subplot(111)
ax1.loglog(pop.wavelengths, pop.data[:,0], color='k', 
    label=r'$t=%i$ Myr' % pop.times[0])
ax1.loglog(pop.wavelengths, pop.data[:,9], color='b', 
        label=r'$t=%i$ Myr' % pop.times[9])

ax1.set_xlabel(ares.util.labels['lambda_AA'])
ax1.set_ylabel(ares.util.labels['intensity_AA'])
ax1.legend(loc='lower right')
ax1.set_ylim(1e25, 1e40)
pl.draw()


fig2 = pl.figure(2); ax2 = fig2.add_subplot(111)
for Z in s99.metallicities.values():
    pop = s99.StellarPopulation(pop_ssp=False, pop_Z=Z)
    
    Nion = pop.PhotonsPerBaryon(13.6, 24.6)
    Nlw = pop.PhotonsPerBaryon(11.2, 13.6)
    ax2.scatter(Nion, Nlw)
    
ax2.set_xlabel(r'$N_{\mathrm{ion}}$')
ax2.set_xlabel(r'$N_{\mathrm{LW}}$')

