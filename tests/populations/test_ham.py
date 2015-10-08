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
from ares.physics.Constants import rhodot_cgs

b15 = ares.util.read_lit('bouwens2015')

lf_pars = b15.lf_pars.copy()
lf_pars['z'] = b15.redshifts

pars = \
{
 'pop_model': 'ham',
 'pop_Macc': 'mcbride2009',
 'pop_constraints': lf_pars,  # bouwens2015
 'pop_yield': 1. / 1.15e-28,
 'pop_yield_units': 'erg/s/Hz/sfr',
 'pop_logM': np.arange(10, 13.5, 0.25), # Masses for AM
 'hmf_func': 'ST',
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
fst = pop.fstar(z=6., M=1e12)
t2 = time.time()

print "Abundance match took %g seconds" % (t2 - t1)

colors = ['k', 'b', 'r', 'g']
for i, z in enumerate(pop.constraints['z']):
    ax2.scatter(pop._Marr, pop._fstar_ham[i], 
        label=r'$z=%g$' % z, color=colors[i], marker='o', facecolors='none')
    
ax2.set_xlabel(r'$M_h / M_{\odot}$')
ax2.set_ylabel(r'$f_{\ast}$')
pl.legend(ncol=2, frameon=False, fontsize=16, loc='lower left')

Marr = np.logspace(8, 14)
for i, z in enumerate(pop.constraints['z']):
    fast = pop.fstar(z=z, M=Marr)
    ax2.loglog(Marr, fast, color=colors[i])

# SFR
fig3 = pl.figure(3); ax3 = fig3.add_subplot(111)
M6 = np.argmin(np.abs(pop.halos.M - 1e6))
for i, z in enumerate(pop.constraints['z']):
    iz = np.argmin(np.abs(pop.halos.z - z))
    ax3.loglog(pop.halos.M[M6:], pop._sfr_ham[iz,M6:])

ax3.set_xlabel(r'$M_h / M_{\odot}$')   
ax3.set_ylabel(r'$\dot{\rho}_{\ast} \ \left[M_{\odot} / \mathrm{yr} \right]$')

# SFRD
fig4 = pl.figure(4); ax4 = fig4.add_subplot(111)
pl.semilogy(pop.halos.z, pop._sfrd_ham_tab, label='HAM')

# Compare to fcoll
pop_fcoll = ares.populations.GalaxyPopulation()
zarr = np.arange(4, 40)
sfrd = np.array(map(pop_fcoll.SFRD, zarr)) * rhodot_cgs
pl.semilogy(zarr, sfrd, color='b', label='fcoll')

# Compare to R15
pop_r15 = ares.util.read_lit('robertson2015')
pl.semilogy(zarr, map(pop_r15.SFRD, zarr), color='g', label='R15')

ax4.set_xlabel(r'$z$')
ax4.set_ylabel(r'$\dot{\rho}_{\ast} \ \left[M_{\odot} / \mathrm{yr} / \mathrm{cMpc}^3 \right]$')
ax4.set_xlim(4, 30)
ax4.set_ylim(1e-10, 1)
pl.legend(loc='lower left', frameon=False)

pl.draw()

