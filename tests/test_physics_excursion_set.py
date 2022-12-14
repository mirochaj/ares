"""

test_excursion_set.py

Author: Jordan Mirocha
Affiliation: McGill
Created on: Mon 18 Feb 2019 10:40:28 EST

Description:

"""

import ares
import numpy as np

def test(tol=0.1, redshifts=[5,10,20]):

    ##
    # Initiailize stuff
    pop = ares.populations.GalaxyPopulation(halo_mf='PS', halo_zmin=5,
        halo_zmax=30)

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
    # Fig. 1. Get the power spectrum
    ##

    ls = '-', '--', ':'
    for i, z in enumerate(redshifts):
        iz = np.argmin(np.abs(z - pop.halos.tab_z))

        k = pop.halos.tab_k_lin
        Delsq = k**3. * pop.halos.tab_ps_lin[iz,:] / (2 * np.pi**2)

    ##
    # Fig. 2. Get the integrand of the variance
    ##
    for _R in [1e-3, 1e-2, 1e-1, 1, 10]:
        W = xset.WindowFourier(k, R=_R)

    ##
    # Fig. 3. Get the variance
    ##

    R = np.logspace(-6, 6, 10000)
    M = xset.Mass(R)
    S = np.array([xset.Variance(0.0, RR) for RR in R])

    ##
    # Fig. 4. Get dndm
    ##
    for i, z in enumerate(redshifts):

        iz = np.argmin(np.abs(z - pop.halos.tab_z))
        Rarr = np.logspace(-6, 6, 10000)
        R, M, dndm = xset.SizeDistribution(z, Rarr)

        dndm_rg = 10**np.interp(np.log10(pop.halos.tab_M),
            np.log10(M), np.log10(dndm))

        rerr = np.abs((pop.halos.tab_dndm[iz,:] - dndm_rg) \
            / pop.halos.tab_dndm[iz,:])

        err_mean = np.nanmean(rerr)
        assert err_mean < tol, \
            "HMF disagreement at >= 1% level for z={}! {:.4f}".format(z, err_mean)



if __name__ == '__main__':
    test()
