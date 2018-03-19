"""

plot_seds.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Thu May 11 13:30:52 CDT 2017

Description: 

"""

import ares
import numpy as np
import matplotlib.pyplot as pl

mass = 10

ls = [':', '--', '-']
colors = ['m', 'c', 'b', 'g', 'k']

for i, alpha in enumerate([-1.5, -1, -0.5]):
    for j, fsc in enumerate([0.1, 0.5, 0.9]):
        
        k = i * 3 + j
        
        fn = 'simpl_M_{0}_fsc_{1:.2f}_alpha_{2:.2f}.txt'.format(mass, fsc, alpha)
        
        x, y = np.loadtxt(fn, unpack=True)
        
        pl.loglog(x, y, color=colors[i], ls=ls[j])
        
# A pure MCD        
mcd = ares.sources.BlackHole(source_sed='mcd', source_mass=10,
    source_Emin=1, source_Emax=5e4, source_EminNorm=500, source_EmaxNorm=8e3)
pl.loglog(x, list(map(mcd.Spectrum, x)), color='k', lw=3)        

# A few power-laws
for logNHI in [20, 21, 22]:
    src = ares.sources.BlackHole(source_sed='pl', source_mass=10,
       source_Emin=1, source_Emax=5e4, source_EminNorm=500, source_EmaxNorm=8e3,
       source_alpha=-1.5, source_logN=logNHI)
      
    pl.loglog(x, list(map(src.Spectrum, x)), color='k', ls='--', lw=3)   
        
pl.xlim(1e2, 3e4)
pl.ylim(1e-6, 5e-3)
pl.xlabel(r'$h\nu / \mathrm{eV}$')
pl.ylabel(r'$I_{\nu}$')
pl.savefig('mcd_vs_simpl.png')

