"""

test_physics_cosmology_ccl.py

Author: Jordan Mirocha
Affiliation: McGill University
Created on: Sat 15 Aug 2020 15:03:04 EDT

Description:

"""

import ares
import numpy as np
import matplotlib.pyplot as pl
from ares.physics.Constants import cm_per_mpc

def test():

    from ares.physics.CosmologyCCL import CosmologyCCL

    # Initialize ARES cosmology instances.
    cosm_ares = ares.physics.Cosmology(cosmology_package=None)
    cosm_ccl = ares.physics.Cosmology(cosmology_package='ccl')

    ##
    # Compare some basic cosmology stuff, e.g., co-moving distance
    fig1, axes1 = pl.subplots(2, 1, num=1)

    zarr = np.arange(4, 30, 0.01)

    d1 = np.array([cosm_ares.ComovingRadialDistance(0, z) for z in zarr])
    d2 = np.array([cosm_ccl.ComovingRadialDistance(0, z) for z in zarr])

    axes1[0].plot(zarr, d1 / cm_per_mpc, color='k', ls='-', label='ares', lw=1)
    axes1[0].plot(zarr, d2 / cm_per_mpc, color='b', ls='--', label='ccl', lw=3)
    axes1[0].set_ylabel(r'$d(z)$ (Mpc)')
    axes1[0].legend(loc='lower right')

    axes1[1].semilogy(zarr, np.abs(d2 - d1) / d1, ls='-')
    axes1[1].set_xlabel(r'$z$')
    axes1[1].set_ylabel('rel err')

    pl.figure(1)
    pl.savefig('hmf_v_ccl_dofz.png')

    ##
    # Compare HMF and power spectra
    pop_ares = ares.populations.HaloPopulation(hmf_package='hmf', hmf_model='Tinker10',
        cosmology_package=None, hmf_load=False, hmf_dz=1,
        hmf_logMmin=6, hmf_logMmax=18)
    pop_ccl = ares.populations.HaloPopulation(hmf_package='ccl', hmf_model='Tinker10',
        cosmology_package='ccl', hmf_load=False, hmf_dz=1,
        hmf_logMmin=6, hmf_logMmax=16)

    ##
    # First: linear matter power spectrum
    fig2, axes2 = pl.subplots(2, 1, num=2)

    zplot = [4, 6, 10, 20]

    #karr = np.logspace(-3, 3, 500)

    colors = 'k', 'b', 'c', 'm'
    for i, z in enumerate(zplot):

        iz = np.argmin(np.abs(z - pop_ares.halos.tab_z))

        axes2[0].loglog(pop_ares.halos.tab_k_lin, pop_ares.halos.tab_ps_lin[iz,:],
            color=colors[i], ls='-', lw=1, label=r'$z={}$'.format(z))
        axes2[0].loglog(pop_ccl.halos.tab_k_lin, pop_ccl.halos.tab_ps_lin[iz,:],
            color=colors[i], ls='--', lw=3)

        # Interpolate to common k grid.
        ares_ps = pop_ares.halos.tab_ps_lin[iz,:]
        ccl_ps = np.exp(np.interp(np.log(pop_ares.halos.tab_k_lin),
            np.log(pop_ccl.halos.tab_k_lin), np.log(pop_ccl.halos.tab_ps_lin[iz,:])))

        rerr = np.abs(ares_ps - ccl_ps) / ares_ps
        axes2[1].loglog(pop_ares.halos.tab_k_lin, rerr, color=colors[i])


    axes2[0].set_xlim(1e-5, 1e4)
    axes2[1].set_xlim(1e-5, 1e4)
    axes2[0].set_ylim(1e-11, 1e4)
    axes2[1].set_ylim(1e-4, 2)
    axes2[0].legend(loc='lower left')
    axes2[1].set_xlabel(r'$k \ [\mathrm{Mpc}^{-1}]$')
    axes2[1].set_ylabel(r'rel error')
    axes2[0].set_ylabel(r'$P_{\mathrm{lin}}(k)$')
    axes2[0].annotate('hmf (solid)', (0.95, 0.95), ha='right', va='top',
        xycoords='axes fraction')
    axes2[0].annotate('ccl (dashed)', (0.95, 0.85), ha='right', va='top',
        xycoords='axes fraction')

    pl.figure(2)
    pl.savefig('hmf_v_ccl_ps.png')

    ##
    # Now, HMF
    fig3, axes3 = pl.subplots(2, 1, num=3)
    for i, z in enumerate(zplot):

        iz = np.argmin(np.abs(z - pop_ares.halos.tab_z))

        ngtm_ares = pop_ares.halos.tab_ngtm[iz,:]
        ngtm_ccl = pop_ccl.halos.tab_ngtm[iz,:]

        axes3[0].loglog(pop_ares.halos.tab_M, ngtm_ares,
            color=colors[i], label=r'$z={}$'.format(z), ls='-', lw=1)
        axes3[0].loglog(pop_ccl.halos.tab_M, ngtm_ccl,
            color=colors[i], ls='--', lw=3)

        # Interpolate to common Mh grid
        ngtm_interp = 10**np.interp(np.log10(pop_ares.halos.tab_M),
            np.log10(pop_ccl.halos.tab_M), np.log10(ngtm_ccl))

        rerr = np.abs(ngtm_ares - ngtm_interp) / ngtm_ares
        axes3[1].loglog(pop_ares.halos.tab_M, rerr, color=colors[i])

    axes3[0].set_xlim(1e5, 1e15)
    axes3[0].set_ylim(1e-15, 1e4)
    axes3[0].legend(loc='lower left')
    axes3[0].annotate('hmf (solid)', (0.95, 0.95), ha='right', va='top',
        xycoords='axes fraction')
    axes3[0].annotate('ccl (dashed)', (0.95, 0.85), ha='right', va='top',
        xycoords='axes fraction')
    axes3[0].set_ylabel(r'$n(>M_h) \ [\mathrm{cMpc}^{-3}]$')

    axes3[1].set_xlim(1e5, 1e15)
    axes3[1].set_ylim(1e-2, 5)

    axes3[1].set_xlabel(r'$M_h / M_{\odot}$')
    axes3[1].set_ylabel('rel err')

    pl.figure(3)
    pl.savefig('hmf_v_ccl_ngtm.png')

if __name__ == '__main__':
    pass
