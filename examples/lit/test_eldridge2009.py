"""

test_eldridge2009.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Mon Apr 11 11:25:01 PDT 2016

Description: 

"""

import ares
import numpy as np
import matplotlib.pyplot as pl

# Stellar SED at 1,50 Myr
fig1 = pl.figure(1); ax1 = fig1.add_subplot(111)

ls = ['-', '--']
colors = ['k', 'g']
for i, binaries in enumerate([False, True]):
    
    pop = ares.populations.SynthesisModel(pop_sed='eldridge2009',
        pop_binaries=binaries)
        
    if binaries:
        bin = 'binary'
    else:
        bin = 'single'

    for j, t in enumerate([1, 50]):
        
        k = np.argmin(np.abs(pop.times - t))
            
        ax1.loglog(pop.wavelengths, pop.data[:,k], color=colors[j],
            label=r'$t=%i$ Myr, %s' % (pop.times[k], bin), ls=ls[i])
        
ax1.set_xlabel(ares.util.labels['lambda_AA'])
ax1.set_ylabel(ares.util.labels['intensity_AA'])
ax1.legend(loc='lower left', fontsize=14)
ax1.set_ylim(1e35, 1e42)
pl.draw()

# Compare to Leitherer et al. at t=10 Myr
fig2 = pl.figure(2); ax2 = fig2.add_subplot(111)

for j, model in enumerate(['eldridge2009', 'leitherer1999']):
    pop = ares.populations.SynthesisModel(pop_sed=model,
        pop_binaries=False)
 
    k = np.argmin(np.abs(pop.times - 10.))    
    ax2.loglog(pop.wavelengths, pop.data[:,k], color=colors[j],
        label=model)
    
ax2.set_xlabel(ares.util.labels['lambda_AA'])
ax2.set_ylabel(ares.util.labels['intensity_AA'])
ax2.legend(loc='upper right', fontsize=14)
ax2.set_ylim(1e35, 1e42)
pl.draw()    