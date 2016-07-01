"""

test_solver_crt_tau.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Tue May 31 20:42:40 PDT 2016

Description: 

"""

import os
import time
import ares
import numpy as np
import matplotlib.pyplot as pl
from ares.physics.Constants import c, ev_per_hz, erg_per_ev

def test(tol=5e-2):


    alpha = -2.
    beta = -6.
    zi = 10.
    zf = 6.
    
    # Initialize radiation background
    pars = \
    {
     'include_He': 0,
     'approx_He': 0,
     
     'initial_redshift': zi,
     'final_redshift': zf,
     
     'pop_type': 'galaxy',
     'pop_sfrd': lambda z: 0.1 * (1. + z)**beta,  # for analytic solution to work this must be const
     'pop_sfrd_units': 'msun/yr/mpc^3',
     'pop_sed': 'pl',
     'pop_alpha': alpha,
     'pop_Emin': 5e2,
     'pop_Emax': 1e3,
     'pop_EminNorm': 2e2,
     'pop_EmaxNorm': 3e4,
     'pop_logN': -np.inf,
     'pop_solve_rte': True,
     'pop_approx_tau': False,
     'pop_tau_Nz': 1e2,
    }
    
    colors = 'k', 'b'
    for i, include_He in enumerate([0, 1]):
        
        pars['include_He'] = include_He
        pars['approx_He'] = include_He
        pars['pop_approx_tau'] = False
    
        # Create OpticalDepth instance
        igm = ares.solvers.OpticalDepth(**pars)
        
        # Impose an ionization history: neutral for all times
        igm.ionization_history = lambda z: 0.0
        
        # Tabulate tau
        tau = igm.TabulateOpticalDepth()
        igm.save(prefix='tau_test', suffix='pkl', clobber=True)
        
        # Run radiation background calculation
        pars['tau_table'] = 'tau_test.pkl'
        sim_1 = ares.simulations.MetaGalacticBackground(**pars)
        sim_1.run()
        
        os.remove('tau_test.pkl')
        
        # Compare to transparent IGM solution
        pars['pop_approx_tau'] = True
        sim_2 = ares.simulations.MetaGalacticBackground(**pars)
        sim_2.run()
        
        # Grab histories
        z1, E1, f1 = sim_1.get_history(0, flatten=True)
        z2, E2, f2 = sim_2.get_history(0, flatten=True)
        
        # Impose a transparent IGM to check tau I/O
        if i == 0:
            
            igm.ionization_history = lambda z: 1.0
            
            # Tabulate tau
            tau = igm.TabulateOpticalDepth()
            igm.save(prefix='tau_test', suffix='pkl', clobber=True)
            
            pars['tau_table'] = 'tau_test.pkl'
            pars['pop_approx_tau'] = False
            sim_3 = ares.simulations.MetaGalacticBackground(**pars)
            sim_3.run()
        
            z3, E3, f3 = sim_3.get_history(0, flatten=True)
        
            os.remove('tau_test.pkl')
        
            assert np.allclose(f3[-1], f2[-1]), "Problem with tau I/O."
        
        # Compare to analytic solution
        if i == 0:
            # Grab GalaxyPopulation
            E = E1
            pop = sim_2.pops[0]
            
            # Cosmologically-limited solution to the RTE
            # [Equation A1 in Mirocha (2014)]
            f_an = np.array(map(lambda E: pop.Emissivity(zf, E), E))
            f_an *= (1. + zf)**(4.5 - (alpha + beta)) / 4. / np.pi \
                / pop.cosm.HubbleParameter(zf) / (alpha + beta - 1.5)
            f_an *= ((1. + zi)**(alpha + beta - 1.5) - (1. + zf)**(alpha + beta - 1.5))
            f_an *= c * ev_per_hz / E / erg_per_ev
            pl.semilogy(E2, f_an, ls=':', color='k', label='analytic')
            
            label = 'neutral'
            pl.semilogy(E2, f2[-1], ls='--', color=colors[i], label='ionized')
            pl.annotate('H only', (0.05, 0.97), xycoords='axes fraction',
                ha='left', va='top')
                
            # Check analytic solution
            diff = np.abs(f_an - f2[-1]) / f_an
            
            # Loose tolerance in this example since tau table is coarse
            assert diff[0] < tol, \
                "Relative error between analytical and numerical solutions exceeds %.3g." % tol
    
        else:
            label = None
            pl.annotate('H+He', (0.05, 0.9), xycoords='axes fraction',
                ha='left', va='top', color='b')
        
        # Plot solution assuming neutral IGM
        pl.semilogy(E1, f1[-1], ls='-', color=colors[i], label=label)
        
        # Assert that transparent IGM -> larger fluxes in soft X-rays
        assert np.all(f2[-1] >= f1[-1])
            
        # Make sure X-ray background is harder when helium is included
        if i == 0:
            f_H = f1[-1].copy()
        else:
            assert np.all(f1[-1] <= f_H), "XRB should be harder when He is included!"
            
    pl.xlabel(ares.util.labels['E'])
    pl.ylabel(ares.util.labels['flux'])
    pl.legend(fontsize=14)
    pl.savefig('%s.png' % __file__.rstrip('.py'))
    pl.close()

if __name__ == '__main__':
    test()
    
   
