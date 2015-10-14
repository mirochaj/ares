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
from ares.physics.Constants import rhodot_cgs, cm_per_mpc

b15 = ares.util.read_lit('bouwens2015')

lf_pars = b15.lf_pars.copy()
lf_pars['z'] = b15.redshifts

pars = \
{
 'pop_model': 'ham',
 'pop_Macc': 'mcbride2009',
 'pop_constraints': lf_pars,  # bouwens2015
 'pop_kappa_UV': 1.15e-28,
 'pop_logM': np.arange(10, 13.5, 0.25), # Masses for AM
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

zarr = np.arange(4, 40)
sfrd_1 = np.array(map(pop.SFRD, zarr)) * rhodot_cgs
ax4.semilogy(zarr, sfrd_1, label='HAM')

# Compare to fcoll
pop_fcoll = ares.populations.GalaxyPopulation()

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

# Halo mass - halo luminosity relationship
fig5 = pl.figure(5); ax5 = fig5.add_subplot(111)

# At z=6
for z in [6,7,8,9]:

    fast = pop.fstar(z=z, M=pop.halos.M)
    iz = np.argmin(np.abs(z - pop.halos.z))
    ax5.loglog(pop.halos.M, pop._sfr_ham[iz,:] / pop.pf['pop_kappa_UV'],
        label=r'$z=%i$' % z)

ax5.set_xlabel(r'$M_h / M_{\odot}$')  
ax5.set_ylabel(r'$L_h / (\mathrm{erg} \ \mathrm{s}^{-1} \ \mathrm{Hz}^{-1})$')  
ax5.set_xlim(1e8, 1e14)
ax5.set_ylim(1e25, 1e31)
ax5.legend(loc='upper left', frameon=False, fontsize=16)

# Ionization emissivity evolution
fig6 = pl.figure(6); ax6 = fig6.add_subplot(111)

Nion = 4000. * np.array(map(pop.SFRD, zarr)) * cm_per_mpc**3 / pop.cosm.g_per_baryon
ax6.semilogy(zarr, Nion, label='HAM')

ax6.set_ylabel(r'$N_{\mathrm{ion}} / (\mathrm{s}^{-1} \ \mathrm{cMpc}^{-3})$')
ax6.set_xlabel(r'$z$')

# Cumulative luminosity density
fig7 = pl.figure(7); ax7 = fig7.add_subplot(111)

from scipy.integrate import cumtrapz

for z in [6,7,8,9]:

    fast = pop.fstar(z=z, M=pop.halos.M)
    iz = np.argmin(np.abs(z - pop.halos.z))
    to_int = pop._sfr_ham[iz,:] * pop.halos.dndm[iz,:] / pop.pf['pop_kappa_UV']
    rhoL = cumtrapz(to_int, x=pop.halos.logM) \
        / np.trapz(to_int, x=pop.halos.logM)
    
    ax7.loglog(pop.halos.M[1:], rhoL, label=r'$z=%i$' % z)

ax7.set_xlabel(r'$M_h / M_{\odot}$')  
ax7.set_ylabel(r'$L_h / (\mathrm{erg} \ \mathrm{s}^{-1} \ \mathrm{Hz}^{-1})$')  
ax7.set_xlim(1e8, 1e13)
ax7.set_ylim(1e-2, 1.05)
ax7.legend(loc='upper left', frameon=False, fontsize=16)

pl.draw()


