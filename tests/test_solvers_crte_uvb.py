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
 'pop_sfr_model': 'sfrd-func',
 'pop_sfrd': lambda z: 0.1 * (1. + z)**beta,  # for analytic solution to work this must be const
 'pop_sfrd_units': 'msun/yr/mpc^3',
 'pop_sed': 'pl',
 'pop_alpha': alpha,
 'pop_Emin': 1.,
 'pop_Emax': 1e2,
 'pop_EminNorm': 13.6,
 'pop_EmaxNorm': 1e2,
 'pop_rad_yield': 1e57,
 'pop_rad_yield_units': 'photons/msun',

 # Solution method
 "lya_nmax": 8,
 'pop_solve_rte': True,
 'tau_redshift_bins': 400,

 'initial_redshift': 40.,
 'final_redshift': 10.,
}

def test(tol=1e-2):

    # First calculation: no sawtooth
    mgb = ares.simulations.MetaGalacticBackground(**pars)
    mgb.run()

    z, E, flux = mgb.get_history(flatten=True)

    Jnu = flux[0] * E * erg_per_ev

    # Grab GalaxyPopulation
    pop = mgb.pops[0]

    # Cosmologically-limited solution to the RTE
    # [Equation A1 in Mirocha (2014)]
    zi, zf = 40., 10.
    e_nu = np.array([pop.Emissivity(zf, EE) for EE in E])
    e_nu *= (1. + zf)**(4.5 - (alpha + beta)) / 4. / np.pi \
        / pop.cosm.HubbleParameter(zf) / (alpha + beta - 1.5)
    e_nu *= ((1. + zi)**(alpha + beta - 1.5) - (1. + zf)**(alpha + beta - 1.5))
    e_nu *= c * ev_per_hz

    # Compare to analytic solution
    flux_anl = e_nu
    flux_num = flux[0] * E * erg_per_ev

    diff = np.abs(flux_anl - flux_num) / flux_anl

    assert diff[0] < tol, \
        "Relative error between analytical and numerical solutions exceeds {:.3g}.".format(tol)


    k = np.argmin(np.abs(E - E_LyA))
    Ja = flux[:,k] * E[k] * erg_per_ev
    Ja_anl = e_nu[k]

    # Compare to case where line cascade is included
    mgb = ares.simulations.MetaGalacticBackground(**pars)
    mgb.run()

    z, E, flux = mgb.get_history(flatten=True)

    Jnu_cas = flux[:,k] * E[k] * erg_per_ev
    

if __name__ == '__main__':
    test()
