"""

test_populations_bh.py

Author: Jordan Mirocha
Affiliation: McGill
Created on: Fri  3 Apr 2020 12:54:41 EDT

Description: 

"""

import ares
import numpy as np
import matplotlib.pyplot as pl
from ares.physics.Constants import rhodot_cgs, rho_cgs

def test():
    bh_pars = \
    {
     'pop_sfr_model': 'bhmd',
     'pop_Tmin': 2e3,
     'pop_Tmax': 1e4,
     'pop_bh_seed_eff': 1e-5,
     'pop_eta': 0.1,
     'pop_fduty': 1.,
     'pop_rad_yield': 0.1, # in this case, fraction of Lbol in (EminNorm, EmaxNorm)
     'pop_Emin': 2e2,
     'pop_Emax': 3e4,
     'pop_EminNorm': 2e3,
     'pop_EmaxNorm': 1e4,
     'pop_ion_src_cgm': False,
     'pop_solve_rte': True,
     'pop_sed': 'pl',
     'pop_alpha': -1.5,
     'sam_dz': 0.05,
     'sam_atol': 1e-6,
     'sam_rtol': 1e-8,
    }
    
    pop = ares.populations.GalaxyPopulation(**bh_pars)
    
    zarr = np.arange(5, 40)
    
    frd = pop.FRD(zarr) * rhodot_cgs
    ard = pop.ARD(zarr) * rhodot_cgs
    bhmd = pop.BHMD(zarr) * rho_cgs
    
    pl.figure(1)
    pl.semilogy(zarr, frd, color='k', label='new BHs')
    pl.semilogy(zarr, ard, color='b', ls='--', label='accretion')
    pl.xlabel(r'$z$')
    pl.ylabel(r'$\dot{\rho}_{\bullet} \ [M_{\odot} \ \mathrm{yr}^{-1} \ \mathrm{cMpc}^{-3}]$')
    pl.legend(loc='lower left', fontsize=14)
    pl.savefig('rho_acc.png')
    
    pl.figure(2)
    pl.semilogy(zarr, bhmd, color='b', ls='--')
    pl.xlabel(r'$z$')
    pl.ylabel(r'$\rho_{\bullet} \ [M_{\odot} \ \mathrm{cMpc}^{-3}]$')

    # Crude checks
    assert 1e-9 <= np.mean(frd) <= 1e-1, "BH FRD unreasonable!"
    assert 1e2 <= np.interp(10, zarr, bhmd) <= 1e9, "BHMD unreasonable!"
    
    
if __name__ == '__main__':
    test()
    
