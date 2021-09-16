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
from ares.physics.Constants import erg_per_ev, c, cm_per_mpc

# Arbitrary units for emissivity, really
SFRD = 1.                  # Msun / yr / cMpc^3
ryield = 1.                # erg / s / (Msun / yr) / Hz
E0 = 1e4  # eV

# Test problem

pars = \
{
 'pop_sfr_model': 'sfrd-func',
 'pop_type': 'galaxy',
 'pop_sfrd': lambda z: SFRD, # Units = Msun / yr / cMpc^3
 'pop_sed': 'delta',
 'pop_Emin': 1e3,
 'pop_Emax': E0,
 'pop_Enorm': E0,

 'pop_rad_yield': ryield,    # 1 erg / s / (Msun / yr)
 'pop_rad_yield_units': 'erg/s/sfr/hz',

 'pop_solve_rte': True,
 'tau_redshift_bins': 1000,
 'tau_approx': True,
 'tau_Emin_pin': False,      # Emax is highest in grid, not Emin

 'initial_redshift': 60.,
 'first_light_redshift': 60.,
 'final_redshift': 6.,
}

def test(tol=1e-2):

    # Analytic solution
    cosm = ares.physics.Cosmology()
    H0 = cosm.hubble_0
    Om = cosm.omega_m_0

    # A = \hat{\epsilon}_{\nu} = SFRD * L * delta(nu - nu0)
    # [A] = photons / s / cm^3 / Hz
    A = SFRD * ryield / cm_per_mpc**3
    A = A / (E0 * erg_per_ev) # convert to photon number
    # Flux in photon number
    J = lambda z, E: (E / E0)**1.5 * (c / 4. / np.pi) * (A / H0 / np.sqrt(Om)) \
        * (1. + z)**0.5

    # I'm off by like a factor of 6.6! It depends on tau_redshift_bins, and does
    # asymptote to a match as tau_redshift_bins grows!

    # Numerical solutions
    mgb = ares.simulations.MetaGalacticBackground(**pars)
    mgb.run()

    z, E, flux = mgb.get_history()

    for j, redshift in enumerate([6, 10, 20, 30]):

        iz = np.argmin(np.abs(redshift - z))

        # Plot up background flux
        f1 = flux[iz][0]

        fanl = J(z[iz], E[0])

        # Plot the errors
        err = np.abs(fanl - f1) / f1

        # Make sure numerical solution accurate to 1%.
        # Must filter out infs since the whole energy space won't get filled
        # since there hasn't been enough time for photons to redshift down
        # to Emin by the end of the calculation.
        assert np.all(err[np.isfinite(err)] < tol)


if __name__ == '__main__':
    test()
