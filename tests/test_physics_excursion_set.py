"""

test_excursion_set.py

Author: Jordan Mirocha
Affiliation: McGill
Created on: Mon 18 Feb 2019 10:40:28 EST

Description: 

"""

import ares
import numpy as np
import matplotlib.pyplot as pl

def test(tol=0.1, redshifts=[5,10,20]):

    ##
    # Initiailize stuff
    pop = ares.populations.GalaxyPopulation(hmf_model='PS', hmf_zmin=5, 
        hmf_zmax=30)
    
    xset_pars = \
    {
     'xset_window': 'tophat-real',
     'xset_barrier': 'constant',
     'xset_pdf': 'gaussian',
    }
    
    xset = ares.physics.ExcursionSet(**xset_pars)
    xset.tab_M = pop.halos.tab_M
    xset.tab_sigma = pop.halos.tab_sigma
    xset.tab_ps = pop.halos.tab_ps_lin
    xset.tab_z = pop.halos.tab_z
    xset.tab_k = pop.halos.tab_k_lin
    xset.tab_growth = pop.halos.tab_growth
    
    ##
    # Fig. 1. Plot the power spectrum
    ##
    
    fig1 = pl.figure(1); ax1 = fig1.add_subplot(111)
    
    ls = '-', '--', ':'
    for i, z in enumerate(redshifts):
        iz = np.argmin(np.abs(z - pop.halos.tab_z))
        
        k = pop.halos.tab_k_lin
        Delsq = k**3. * pop.halos.tab_ps_lin[iz,:] / (2 * np.pi**2)
        ax1.loglog(k, Delsq, color='k', ls=ls[i], label=r'$z={}$'.format(z))
        
    ax1.set_xlim(1e-3, 1e3)
    ax1.set_ylim(1e-6, 1e3)
    ax1.set_xlabel(r'$k \ [\mathrm{cMpc}^{-1}]$')
    ax1.set_ylabel(r'$k^3 P_{\mathrm{lin}}(k) / 2 \pi^2$')
    ax1.legend(loc='lower right')
    
    
    
    ##
    # Fig. 2. Plot the integrand of the variance
    ##
    
    fig2 = pl.figure(2); ax2 = fig2.add_subplot(111)
    
    for _R in [1e-3, 1e-2, 1e-1, 1, 10]:
        W = xset.WindowFourier(k, R=_R)
        ax2.loglog(k, Delsq * np.abs(W)**2, color='k')
    ax2.set_xlabel(r'$k$')
    ax2.set_ylabel(r'$\Delta(k)^2 |W(k)|^2$')
    
    ##
    # Fig. 3. Plot the variance
    ##
    
    fig3 = pl.figure(3); ax3 = fig3.add_subplot(111)
    
    R = np.logspace(-6, 6, 10000)
    M = xset.Mass(R)
    S = np.array([xset.Variance(0.0, RR) for RR in R])
    
    ax3.loglog(pop.halos.tab_M, pop.halos.tab_sigma**2, color='k')
    ax3.loglog(M, S, color='b', ls='--', lw=3, alpha=0.7)
    ax3.set_xlabel(r'$M / M_{\odot}$')
    ax3.set_ylabel(r'$\sigma^2(M)$')
    ax3.set_xlim(1e4, 1e18)
    ax3.set_ylim(1e-2, 1e3)
    
    ##
    # Fig. 4. Plot dndm
    ##
    fig4 = pl.figure(4); ax4 = fig4.add_subplot(111)
    
    for i, z in enumerate(redshifts):
    
        iz = np.argmin(np.abs(z - pop.halos.tab_z))
        ax4.loglog(pop.halos.tab_M, pop.halos.tab_dndm[iz,:], color='k', ls=ls[i],
            label=r'$z={}$'.format(z))
    
        Rarr = np.logspace(-6, 6, 10000)
        R, M, dndm = xset.SizeDistribution(z, Rarr)
        ax4.loglog(M, dndm, color='b', ls=ls[i], lw=3, alpha=0.7)
        
        dndm_rg = 10**np.interp(np.log10(pop.halos.tab_M), 
            np.log10(M), np.log10(dndm))
            
        rerr = np.abs((pop.halos.tab_dndm[iz,:] - dndm_rg) \
            / pop.halos.tab_dndm[iz,:])
        
        err_mean = np.nanmean(rerr)
        assert err_mean < tol, \
            "HMF disagreement at >= 1% level for z={}! {:.4f}".format(z, err_mean)
    
    
    ax4.annotate(r'via hmf', (0.05, 0.15), xycoords='axes fraction', 
        ha='left', va='top')
    ax4.annotate(r'via ARES excursion set', (0.05, 0.1), xycoords='axes fraction',    
        ha='left', va='top', color='b')
    ax4.set_xlabel(r'$M_h / M_{\odot}$')
    ax4.set_ylabel(r'$dn/dm$')
    ax4.set_xlim(1e6, 1e17)
    ax4.set_ylim(1e-35, 1e1)
    ax4.legend(loc='upper right', fontsize=12)
    
    
if __name__ == '__main__':
    test()
    