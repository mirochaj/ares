"""

test_abundance_matching.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Thu Jan 21 12:41:03 PST 2016

Description: Show how varying the UV slope (redshift and magnitude 
independent) changes the inferred star formation efficiency when doing
abundance matching.

"""

import ares
import numpy as np
import matplotlib.pyplot as pl

# 
## INPUT
redshifts = [3.8, 4.9, 5.9, 6.9]
betas = [None, -2]
method = [None, 'meurer1999']
##
#

markers = ['o', 's', '^', '>']

# Masses
M = np.logspace(8, 14)

# For redshifts
colors = 'k', 'b', 'r', 'g', 'c', 'm'

for h, beta in enumerate(betas):
    
    if h == 0:
        continue

    pars = {'dustcorr_Afun': method[h], 'dustcorr_Bfun_par0': beta}

    for i, z in enumerate(redshifts):
        
        ham = ares.inference.AbundanceMatching(sfe_Mfun='dpl', pop_model='precip', 
            pop_kappa_UV=1e-29, **pars)

        ham.redshifts = [z]
        ham.constraints = 'bouwens2015'

        # Fit it real quick
        best = ham.fit_fstar()
        
        if h > 0:
            label = r'$z=%.2g, \beta=%.2g$' % (z, beta)
        else:
            label = 'no DC'
            
        pl.scatter(ham.MofL_tab[0], ham.fstar_tab[0], color=colors[i],
            label=label, marker=markers[h], facecolors='none', s=50)

        pl.loglog(M, ham.fstar._call(z, M, best), color=colors[i])

pl.xscale('log')
pl.yscale('log')
pl.xlabel(r'$M_h / M_{\odot}$')
pl.ylabel(r'$f_{\ast}$')
pl.legend(loc='lower left', ncol=1, fontsize=14)





