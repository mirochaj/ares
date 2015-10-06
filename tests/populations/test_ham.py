"""

test_empirical.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Sep  2 19:56:36 PDT 2015

Description: 

"""

import ares
import numpy as np
import matplotlib.pyplot as pl

b15 = ares.util.read_lit('bouwens2015')

lf_pars = b15.lf_pars.copy()
lf_pars['z'] = b15.redshifts

pars = \
{
 'pop_model': 'ham',
 'pop_Macc': 'mcbride2009',
 'pop_constraints': lf_pars,
 'pop_yield': 1. / 1.15e-28,
 'pop_yield_units': 'erg/s/Hz/sfr',
}

pop = ares.populations.GalaxyPopulation(**pars)

fig1 = pl.figure(1); ax1 = fig1.add_subplot(111)

L = np.logspace(27., 30.)
for i, z in enumerate(pop.constraints['z']):
    phi = pop.constraints['phi_star'][i] 
    phi *= (L / pop.constraints['L_star'][i])**pop.constraints['alpha'][i]
    phi *= np.exp(-L / pop.constraints['L_star'][i])
    
    ax1.loglog(L, phi)
    
ax1.set_xlabel(r'$L \ (\mathrm{erg} \ \mathrm{s}^{-1}) \ \mathrm{Hz}^{-1}$')
ax1.set_ylabel(r'$\phi(L)$')

# SFE
fig2 = pl.figure(2); ax2 = fig2.add_subplot(111)

import time

t1 = time.time()
fst = pop.fstar
t2 = time.time()

print "Abundance match took %g seconds" % (t2 - t1)

for i, z in enumerate(pop.constraints['z']):
    ax2.loglog(pop._Marr, pop.fstar[i], label=r'$z=%g$' % z)
    
ax2.set_xlabel(r'$M_h / M_{\odot}$')
ax2.set_ylabel(r'$f_{\ast}$')
pl.legend(ncol=2, frameon=False, fontsize=16)
