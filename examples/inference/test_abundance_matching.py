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
redshifts = [4.9]
betas = [None, -2]
method = [None, 'meurer1999']
##
#
markers = ['o', 's', '^', '>']

# For redshifts
colors = 'k', 'b', 'r', 'g', 'c', 'm'

for h, beta in enumerate(betas):

    pars = {'dustcorr_Afun': method[h], 'dustcorr_Bfun_par0': beta}

    ham = ares.inference.AbundanceMatching(**pars)

    ham.constraints = 'bouwens2015'
    ham.redshifts = redshifts

    if h > 0:
        label = r'$z=%.2g, \beta=%.2g$' % (z, beta)
    else:
        label = 'no DC'

    for i, z in enumerate(ham.redshifts):
        pl.scatter(ham.MofL_tab[i], ham.fstar_tab[i], color=colors[i],
            label=label, marker=markers[h])
    
pl.xscale('log')
pl.yscale('log')
pl.xlabel(r'$M_h / M_{\odot}$')
pl.ylabel(r'$f_{\ast}$')
pl.legend(loc='lower left', ncol=1, fontsize=14)



