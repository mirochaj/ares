"""

test_sed_mcd.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Thu May 11 09:39:08 CDT 2017

Description: 

"""

import ares
import time
import numpy as np
import matplotlib.pyplot as pl

mcd = \
{
 'source_type': 'bh', 
 'source_mass': 10,
 'source_rmax': 1e2,
 'source_sed': 'mcd',
 'source_Emin': 10,
 'source_Emax': 5e4,
 'source_EminNorm': 500.,
 'source_EmaxNorm': 8e3,
 'source_alpha': -1.5,
 'source_fsc': 0.9,
}

simpl = mcd.copy()
simpl['source_sed'] = 'simpl'

E = np.logspace(0, 4.5)

src = ares.sources.BlackHole(**mcd)
pl.loglog(E, list(map(src.Spectrum, E)), color='k')

ls = [':', '--', '-']
colors = ['m', 'c', 'b', 'k']
for i, alpha in enumerate([-1, 0., 0.5]):
    for j, fsc in enumerate([0.1, 0.5, 0.9]):
        
        simpl['source_alpha'] = alpha
        simpl['source_fsc'] = fsc
        
        src2 = ares.sources.BlackHole(**simpl)
        
        t1 = time.time()
        pl.loglog(E, list(map(src2.Spectrum, E)), color=colors[j], ls=ls[i])
        t2 = time.time()
        print('simpl took {:.2g} sec'.format(t2 - t1))
    
pl.xlim(10, 3e4)
pl.ylim(1e-6, 1e-3)
pl.xlabel(r'$h\nu / \mathrm{eV}$')
pl.ylabel(r'$I_{\nu}$')
pl.savefig('mcd_vs_simpl.png')



