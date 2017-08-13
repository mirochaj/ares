"""

test_generator_lwb.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri Aug 16 12:59:40 MDT 2013

Description: This is very similar to Haiman, Abel, & Rees (1997) Fig. 1.

"""

import ares
import numpy as np
import matplotlib.pyplot as pl
from ares.physics.Constants import erg_per_ev, c, ev_per_hz, E_LyA

beta = -6.
alpha = 0.

pars = \
{
 'pop_type': 'galaxy',
 'pop_sfr_model': 'sfrd-func',
 'pop_sfrd': lambda z: 0.1 * (1. + z)**beta,  # for analytic solution to work this must be const
 'pop_sfrd_units': 'msun/yr/mpc^3',
 'pop_sed': 'pl',
 'pop_alpha': alpha, 
 'pop_Emin': 1.,
 'pop_Emax': 1e2,
 'pop_EminNorm': 11.2,
 'pop_EmaxNorm': 13.6,
 'pop_rad_yield': 1e4,
 'pop_rad_yield_units': 'photons/baryon',

 # Solution method
 "lya_nmax": 8,
 'pop_solve_rte': True,
 'pop_tau_Nz': 400,

 'initial_redshift': 40.,
 'final_redshift': 10.,
}

pars2 = pars.copy()
pars2['pop_sed'] = 'eldridge2009'
pars2['pop_rad_yield'] = 'from_sed'

f1 = pl.figure(1); ax1 = f1.add_subplot(111)
f2 = pl.figure(2); ax2 = f2.add_subplot(111)

colors = 'k', 'b'
for i, p in enumerate([pars, pars2]):

    # First calculation: no sawtooth
    mgb = ares.simulations.MetaGalacticBackground(**p)
    mgb.run()
    
    z, E, flux = mgb.get_history(flatten=True)
        
    ax1.semilogy(E, flux[0] * E * erg_per_ev, color=colors[i], ls='--')
    
    # Grab GalaxyPopulation
    pop = mgb.pops[0]
    
    
    # Plot Lyman alpha flux evolution
    k = np.argmin(np.abs(E - E_LyA))
    
    ax2.semilogy(z, flux[:,k] * E[k] * erg_per_ev, color=colors[i])
    
    # Compare to case where line cascade is included
    mgb = ares.simulations.MetaGalacticBackground(include_injected_lya=True, **p)
    mgb.run()
    
    z, E, flux = mgb.get_history(flatten=True)
    ax2.semilogy(z, flux[:,k] * E[k] * erg_per_ev, color=colors[i], ls='--')
    
    # Analytic solutions
    #if i > 0:
    #    continue
    
    # Cosmologically-limited solution to the RTE
    # [Equation A1 in Mirocha (2014)]
    zi, zf = 40., 10.
    e_nu = np.array(map(lambda E: pop.Emissivity(zf, E), E))
    e_nu *= (1. + zf)**(4.5 - (alpha + beta)) / 4. / np.pi \
        / pop.cosm.HubbleParameter(zf) / (alpha + beta - 1.5)
    e_nu *= ((1. + zi)**(alpha + beta - 1.5) - (1. + zf)**(alpha + beta - 1.5))
    e_nu *= c * ev_per_hz
    
    # Compare to analytic solution
    flux_anl = e_nu
    flux_num = flux[-1] * E * erg_per_ev
    
    diff = np.abs(flux_anl - flux_num) / flux_anl
    
    #assert diff[0] < tol, \
    #    "Relative error between analytical and numerical solutions exceeds %.3g." % tol
    
    # Plot it
    ax1.semilogy(E, e_nu, color=colors[i], ls='-', alpha=0.7)
    
    # Plot analytic solution for Ly-a flux
    ax2.scatter(zf, e_nu[k], s=150, marker='+', color=colors[i])
        
ax1.set_xlabel(ares.util.labels['E'])
ax1.set_ylabel(ares.util.labels['flux_E'])
ax1.set_ylim(1e-30, 1e-24)
ax1.set_xlim(0, 60)    
    
ax2.set_xlabel(r'$z$')
ax2.set_ylabel(r'$J_{\alpha}$')
    
pl.show()    
    
pl.figure(1)
pl.savefig('%s_1.png' % (__file__.rstrip('.py')))

pl.figure(2)
pl.savefig('%s_2.png' % (__file__.rstrip('.py')))

pl.close('all')    
    
    