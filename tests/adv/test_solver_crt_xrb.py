"""

test_generator_xrb.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri Jun 14 09:20:22 2013

Description:

"""

import ares
import numpy as np
import matplotlib.pyplot as pl
from ares.physics.Constants import erg_per_ev, c, ev_per_hz

# Initialize radiation background
pars = \
{
 # Source properties
 'pop_type': 'galaxy',
 'pop_sfrd': lambda z: 0.1,
 'pop_sfrd_units': 'msun/yr/mpc^3', 
 'pop_sed': 'pl',
 'pop_alpha': -1.5,
 'pop_Emin': 1e2,
 'pop_Emax': 3e4,
 'pop_EminNorm': 5e2,
 'pop_EmaxNorm': 8e3,
 'pop_yield': 2.6e39,
 'pop_yield_units': 'erg/s/sfr',

 'pop_solve_rte': True,
 "pop_tau_Nz": 400,

 'initial_redshift': 40.,
 'final_redshift': 10.,
}


def test(Ecomp=8e3, tol=1e-2):

    mgb = ares.simulations.MetaGalacticBackground(**pars)
    mgb.run()
    
    """
    First, look at background flux itself.
    """
    
    z, E, flux = mgb.get_history(flatten=True)
    flux_thin = flux * E * erg_per_ev
    
    mask = slice(0, -1, 50)
    pl.scatter(E[mask], flux_thin[-1][mask], color='b', facecolors='none', s=100)
    
    # Grab GalaxyPopulation
    pop = mgb.pops[0]

    # Cosmologically-limited solution to the RTE
    # [Equation A1 in Mirocha (2014)]
    zi, zf = 40., 10.
    e_nu = np.array([pop.Emissivity(10., EE) for EE in E])
    e_nu *= c / 4. / np.pi / pop.cosm.HubbleParameter(10.) 
    e_nu *= (1. + 10.)**6. / -3.
    e_nu *= ((1. + 40.)**-3. - (1. + 10.)**-3.)
    e_nu *= ev_per_hz

    # Plot it
    pl.loglog(E, e_nu, color='k', ls='-')
    pl.xlabel(ares.util.labels['E'])
    pl.ylabel(ares.util.labels['flux_E'])

    """
    Do neutral absorption in IGM.
    """

    pars['pop_solve_rte'] = True
    pars['pop_approx_tau'] = 'neutral'
    pars['pop_tau_Nz'] = 400
    
    mgb = ares.simulations.MetaGalacticBackground(**pars)
    mgb.run()
    
    z, E2, flux2 = mgb.get_history(flatten=True)
    
    flux_thick = flux2 * E2 * erg_per_ev
    pl.loglog(E2, flux_thick[-1], color='k', ls=':')
    
    pl.ylim(0.5 * e_nu[-1], e_nu[0] * 2)
    
    # Compare results at optically thin energy away from edges    
    flux_comp_anl = e_nu[np.argmin(np.abs(Ecomp - E))]
    flux_comp_thin = flux_thin[-1][np.argmin(np.abs(Ecomp - E))]
    flux_comp_thick = flux_thick[-1][np.argmin(np.abs(Ecomp - E2))]
    
    thin_OK = abs((flux_comp_thin - flux_comp_anl) / flux_comp_anl) \
        < tol
    thick_OK = abs((flux_comp_thick - flux_comp_anl) / flux_comp_anl) \
        < tol    
    
    print("\n# Analytic (thin) ; Numerical (thin) ; Numerical (neutral)")
    print("----------------------------------------------------------")
    print("{0:.8e}    ; {1:.8e}   ; {2:.8e}".format(\
        flux_comp_anl, flux_comp_thin, flux_comp_thick))
    print("----------------------------------------------------------")
    print("relative error    : {0:.12f}   ; {1:.12f}".format(\
        abs((flux_comp_thin - flux_comp_anl) / flux_comp_anl),
        abs((flux_comp_thick - flux_comp_anl) / flux_comp_anl)))
    print("----------------------------------------------------------")
    
    pl.savefig('{!s}.png'.format(__file__.rstrip('.py')))
    pl.close()    

    assert thin_OK and thick_OK, \
        "Relative error between analytical and numerical solutions exceeds {:.3g}.".format(tol)

if __name__ == '__main__':
    test()
   
