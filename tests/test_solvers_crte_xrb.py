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
from ares.physics.Constants import erg_per_ev, c, ev_per_hz, sqdeg_per_std

# Unabsorbed power-law
beta = -6.
alpha = -2.

plpars = \
{
 'pop_sfr_model': 'sfrd-func',
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

 'pop_solve_rte': True,
 'tau_redshift_bins': 400,

 'tau_redshift_bins': 400,
 'initial_redshift': 40.,
 'final_redshift': 10.,

}

def test(tol=1e-2):

    assert alpha + beta != 1.5, "Analytic solution diverges for alpha+beta=3/2!"

    # Absorbed power-law
    aplpars = plpars.copy()
    aplpars.update({'pop_logN': 21., 'pop_hardening': 'extrinsic'})

    # Loop over sources and plot CXRB
    colors = ['k', 'b']
    for i, pars in enumerate([plpars, aplpars]):

        mgb = ares.simulations.MetaGalacticBackground(**pars)
        mgb.run()

        if np.isfinite(mgb.pf['pop_logN']):
            label = r'$N = 10^{{{}}} \ \mathrm{{cm}}^{{-2}}$'.format(int(mgb.pf['pop_logN']))
        else:
            label = r'$N = 0 \ \mathrm{cm}^{-2}$'

        z, E, flux = mgb.get_history()

        # Soft X-ray background
        sxb = mgb.get_spectrum_integrated((5e2, 2e3))

        # Check analytic solution for unabsorbed case
        if i == 0:
            # Grab GalaxyPopulation
            pop = mgb.pops[0]

            # Cosmologically-limited solution to the RTE
            # [Equation A1 in Mirocha (2014)]
            zi, zf = 40., 10.
            e_nu = np.array([pop.get_emissivity(zf, EE) for EE in E[0]])
            e_nu *= (1. + zf)**(4.5 - (alpha + beta)) / 4. / np.pi \
                / pop.cosm.HubbleParameter(zf) / (alpha + beta - 1.5)
            e_nu *= ((1. + zi)**(alpha + beta - 1.5) - (1. + zf)**(alpha + beta - 1.5))
            e_nu *= c * ev_per_hz

            # Compare to analytic solution
            flux_anl = e_nu
            flux_num = flux[0][0] * E[0] * erg_per_ev

            diff = np.abs(flux_anl - flux_num) / flux_anl

            # Only use softest X-ray bin since this is where error should
            # be worst.
            assert diff[0] < tol, \
                "Relative error between analytical and numerical solutions exceeds {:.3g}.".format(tol)

        # Plot up heating rate evolution
        heat = np.zeros_like(z)
        ioniz = np.zeros_like(z)
        for j, redshift in enumerate(z):
            # We have to add brackets because volume.*Rate routines expect
            # fluxes in the form (Npops, Nbands, Nfreq)
            heat[j] = mgb.solver.volume.HeatingRate(redshift, fluxes=[flux[j]])
            ioniz[j] = mgb.solver.volume.IonizationRateIGM(redshift, fluxes=[flux[j]])


if __name__ == '__main__':
    test()
