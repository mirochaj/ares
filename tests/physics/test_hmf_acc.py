"""

test_acc.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Oct 30 09:36:05 MDT 2014

Description: 

"""

import numpy as np
from hmf import MassFunction
import matplotlib.pyplot as pl

ax1 = pl.subplot(211)
ax2 = pl.subplot(212)

mf = MassFunction(Mmin=8., Mmax=16., mf_fit='ST')

colors = 'k', 'b'
for i, z in enumerate([0, 2]):
    
    # Load genmf data
    data = np.loadtxt('data/ST_%i_hmf' % z, unpack=True, usecols=(0, 2))
    M, ngtm = 10**data[0][-1::-1], 10**data[1][-1::-1]
    
    # Update hmf
    mf.update(z=z)
    
    if i == 0:
        labels = 'genmf', 'hmf'
    else:
        labels = [None] * 2
    
    ax1.loglog(M, ngtm, color=colors[i], ls='-', label=labels[0])
    ax1.loglog(mf.M, mf.ngtm, color=colors[i], ls='--', label=labels[1])
    
    # Interpolate each to common mass axis for comparison
    new_M = np.logspace(8, 16, 1e3)

    err = np.abs(np.interp(new_M, M, ngtm) \
               - np.interp(new_M, mf.M, mf.ngtm)) \
               / np.interp(new_M, mf.M, mf.ngtm)
               
    ax2.semilogx(new_M, err, label=r'$z=%i$' % z)

ax2.set_yscale('log')
ax1.legend(loc='upper right')
ax2.set_xlabel(r'$M \ [M_{\odot} / h_{70}]$')
ax2.set_ylabel(r'% error')
ax1.set_ylabel(r'$n(>M) \ [(h_{70} / \mathrm{cMpc})^{3}]$')
ax1.set_ylim(1e-20, 1e3)

ax1.set_xlim(1e8, 1e16)
ax2.set_xlim(1e8, 1e16)
ax1.set_xticklabels([])
ax2.legend(loc='upper left', ncol=2)
ax1.legend(loc='lower left')

pl.draw()

