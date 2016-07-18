"""

test_cxrb_pl.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Dec 19 12:40:52 2012

Description: Compare ionization and heating of background sourced by 
increasingly absorbed power-law X-ray emission.

"""

import ares
import numpy as np
import matplotlib.pyplot as pl
from ares.physics.Constants import erg_per_ev, c, ev_per_hz

# Unabsorbed power-law
beta = -6.
alpha = -2.

plpars = \
{
 'pop_type': 'galaxy',
 'pop_sfrd': lambda z: 0.1 * (1. + z)**beta,  # for analytic solution to work this must be const
 'pop_sfrd_units': 'msun/yr/mpc^3',
 'pop_sed': 'pl',
 'pop_alpha': -2.,
 'pop_Emin': 2e2,
 'pop_Emax': 3e4,
 'pop_EminNorm': 2e2,
 'pop_EmaxNorm': 3e4,
 'pop_logN': -np.inf,
 #'approx_xrb': 0,
 'pop_solve_rte': True,
 'pop_tau_Nz': 400,
 
 'initial_redshift': 40.,
 'final_redshift': 10.,
}

def test(tol = 1e-2):

    assert alpha + beta != 1.5, "Analytic solution diverges for alpha+beta=3/2!"
    
    # Absorbed power-law
    aplpars = plpars.copy()
    aplpars.update({'pop_logN': 21., 'pop_hardening': 'extrinsic'})
    
    fig1 = pl.figure(0); ax1 = fig1.add_subplot(111)
    fig2 = pl.figure(1); ax2 = fig2.add_subplot(111)
    fig3 = pl.figure(2); ax3 = fig3.add_subplot(111)
    
    # Loop over sources and plot CXRB
    colors = ['k', 'b']
    for i, pars in enumerate([plpars, aplpars]):
        
        mgb = ares.simulations.MetaGalacticBackground(**pars)
        mgb.run()
        
        if np.isfinite(mgb.pf['pop_logN']):
            label = r'$N = 10^{%i} \ \mathrm{cm}^{-2}$' % (mgb.pf['pop_logN'])
        else:
            label = r'$N = 0 \ \mathrm{cm}^{-2}$'
            
        z, E, flux = mgb.get_history()
            
        # Plot up background flux
        ax1.loglog(E[0], flux[0][0] * E[0] * erg_per_ev, color=colors[i], ls='-', 
            label=label)
        
        # Check analytic solution for unabsorbed case
        if i == 0:
            # Grab GalaxyPopulation
            pop = mgb.pops[0]
            
            # Cosmologically-limited solution to the RTE
            # [Equation A1 in Mirocha (2014)]
            zi, zf = 40., 10.
            e_nu = np.array(map(lambda E: pop.Emissivity(zf, E), E[0]))
            e_nu *= (1. + zf)**(4.5 - (alpha + beta)) / 4. / np.pi \
                / pop.cosm.HubbleParameter(zf) / (alpha + beta - 1.5)
            e_nu *= ((1. + zi)**(alpha + beta - 1.5) - (1. + zf)**(alpha + beta - 1.5))
            e_nu *= c * ev_per_hz
                    
            ax1.loglog(E[0], e_nu, color='r', ls='--', lw=3, label='analytic')
            
            # Compare to analytic solution
            flux_anl = e_nu
            flux_num = flux[0][0] * E[0] * erg_per_ev
            
            diff = np.abs(flux_anl - flux_num) / flux_anl
            
            assert diff[0] < tol, \
                "Relative error between analytical and numerical solutions exceeds %.3g." % tol
        
        # Plot up heating rate evolution
        heat = np.zeros_like(z)
        ioniz = np.zeros_like(z)
        for j, redshift in enumerate(z):
            # We have to add brackets because volume.*Rate routines expect
            # fluxes in the form (Npops, Nbands, Nfreq)
            heat[j] = mgb.volume.HeatingRate(redshift, fluxes=[flux[j]])
            ioniz[j] = mgb.volume.IonizationRateIGM(redshift, fluxes=[flux[j]])
            
        ax2.semilogy(z, heat, color=colors[i], ls='-', label=label)
        ax3.semilogy(z, ioniz, color=colors[i], ls='-', label=label)
        
    # Make pretty                    
    ax1.set_xlabel(ares.util.labels['E'])
    ax1.set_ylabel(ares.util.labels['flux'])
    ax2.set_xlabel(r'$z$')
    ax2.set_ylabel(r'$\epsilon_X$')
    ax3.set_xlabel(r'$z$')
    ax3.set_ylabel(r'$\Gamma_{\mathrm{HI}}$')
    #ax1.set_ylim(1e-21, 1e-14)
    
    for ax in [ax1, ax2, ax3]:
        ax.legend(loc='best', fontsize=14)
    
    pl.show()
    for i in range(3):
        pl.figure(i)
        pl.savefig('%s_%i.png' % (__file__.rstrip('.py'), i))

    pl.close('all')    
    assert True

if __name__ == '__main__':
    test()

    