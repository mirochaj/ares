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
redshifts = [3.8]
betas = [-2, -2]
s_betas = [0.] * 2
method = ['meurer1999']*2#, None, 'evolving']
fbeta = ['constant', 'FitMason']
##
#

markers = ['o', 's', '^', '>']

# Masses
M = np.logspace(8, 13.2)

# For redshifts
colors = 'k', 'b', 'r', 'g', 'c', 'm'
ls = [':', '--', '-']

#for h, beta in enumerate(betas):
for h, s_beta in enumerate(s_betas):
    
    pars = {'dustcorr_Afun': method[h], 'dustcorr_Bfun': fbeta[h],
        's_beta': s_beta, 'dustcorr_Bfun_par0': betas[h]}

    for i, z in enumerate(redshifts):
                
        ham = ares.inference.AbundanceMatching(php_Mfun='dpl', 
            pop_model='sfe', pop_fstar='php', pop_L1500_per_sfr=3*8.7e27, **pars)

        ham.redshifts = [z]
        ham.constraints = 'bouwens2015'
        
        # Fit it real quick
        best = ham.fit_fstar()
        
        if h > 0:
            #label = r'$z=%.2g, \beta=%.2g$' % (z, beta)
            label = r'$z=%.2g, \sigma_{\beta}=%s$, %s' % (z, s_beta, method[h])
        else:
            #label = 'no DC'
            label = r'$z=%.2g, \sigma_{\beta}=%s$, %s' % (z, s_beta, method[h])
            
        pl.scatter(ham.MofL_tab[0], ham.fstar_tab[0], color=colors[i],
            label=label, marker=markers[h], facecolors='none', s=50)

        pl.loglog(M, ham.fstar._call(z, M, best), color=colors[i], ls=ls[h])

pl.xscale('log')
pl.yscale('log')
pl.xlim(1e8, 2e13)
pl.loglog([1e8, 2e13], [0.5]*2, color='r', ls=':')
pl.loglog([1e8, 2e13], [0.3]*2, color='r', ls=':')
pl.loglog([1e8, 2e13], [0.1]*2, color='r', ls=':')
pl.xlabel(r'$M_h / M_{\odot}$')
pl.ylabel(r'$f_{\ast}$')
pl.legend(loc='lower right', ncol=2, fontsize=10)
pl.show()





