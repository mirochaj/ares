"""

test_s99.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sat Jun  6 11:12:51 MDT 2015

Description: 

"""

import ares
import numpy as np
import matplotlib.pyplot as pl

pop = ares.populations.SynthesisModel(pop_sed='leitherer1999')
    
# Stellar SED at 1,50 Myr
fig1 = pl.figure(1); ax1 = fig1.add_subplot(111)

colors = ['k', 'g']    
for j, t in enumerate([1, 50]):
    k = np.argmin(np.abs(pop.times - t))
    ax1.loglog(pop.wavelengths, pop.data[:,k], color=colors[j],
        label=r'$t={}$ Myr'.format(pop.times[k]))
        
ax1.set_xlabel(ares.util.labels['lambda_AA'])
ax1.set_ylabel(ares.util.labels['intensity_AA'])
ax1.legend(loc='upper right', fontsize=14)
ax1.set_ylim(1e35, 1e42)
pl.draw()

fig2 = pl.figure(2); ax2 = fig2.add_subplot(111)
for Z in pop.metallicities.values():
    pop = ares.populations.SynthesisModel(pop_sed='leitherer1999',
        pop_ssp=False, pop_Z=Z)
    
    Nion = pop.PhotonsPerBaryon(13.6, 24.6)
    Nlw = pop.PhotonsPerBaryon(11.2, 13.6)
    ax2.scatter(Nion, Nlw)
    
ax2.set_xlabel(r'$N_{\mathrm{ion}}$')
ax2.set_xlabel(r'$N_{\mathrm{LW}}$')

# Test interpolation
fig3 = pl.figure(3); ax3 = fig3.add_subplot(111)

pop1 = ares.populations.SynthesisModel(pop_sed='leitherer1999',
    pop_ssp=True, pop_Z=0.001)
pop2 = ares.populations.SynthesisModel(pop_sed='leitherer1999',
    pop_ssp=True, pop_Z=0.04)

ax3.loglog(pop1.wavelengths, pop1.data[:,9], color='k', ls='-')
ax3.loglog(pop1.wavelengths, pop2.data[:,9], color='k', ls='--')

for Z in np.logspace(np.log10(0.002), np.log10(0.02), 3):
    pop = ares.populations.SynthesisModel(pop_sed='leitherer1999',
        pop_ssp=True, pop_Z=Z)
    
    ax3.loglog(pop.wavelengths, pop.data[:,9])
    
ax3.set_xlabel(ares.util.labels['lambda_AA'])
ax3.set_ylabel(ares.util.labels['intensity_AA'])
ax3.legend(loc='lower right')
ax3.set_ylim(1e25, 1e40)
pl.draw()    
