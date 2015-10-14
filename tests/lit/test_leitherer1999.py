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

# Currently, the starburst99 module works a little differently than others. Since there are so many
# "switches", rather than "knobs", it gets its own StellarPopulation class to deal with this. Need
# to work on how to generalize this with the rest of the API.
pop = s99.StellarPopulation(continuous_sf=False, Z=0.04)

# Stellar SED at 1 Myr
fig1 = pl.figure(1); ax1 = fig1.add_subplot(111)
ax1.loglog(pop.wavelengths, pop.data[:,0], color='k')
ax1.loglog(pop.wavelengths, pop.data[:,10], color='b')
ax1.set_ylim(1e30, 1e42)

ax1.set_xlabel(ares.util.labels['lambda_AA'])
ax1.set_ylabel(ares.util.labels['intensity_AA'])

pl.draw()

# Plot Nion / Nlw vs. time
fig2 = pl.figure(2); ax2 = fig2.add_subplot(111)

ax2.loglog(pop.times, pop.PhotonsPerBaryon(10.2, 13.6), color='k', ls='-', label='LW')
ax2.loglog(pop.times, pop.PhotonsPerBaryon(13.6, 24.4), color='k', ls='--', label='LyC')
ax2.legend(loc='upper left')
ax2.set_xlabel(ares.util.labels['t_myr'])
ax2.set_ylabel('photons / baryon')
pl.draw()

