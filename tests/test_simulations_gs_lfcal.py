"""

test_simulations_gs_lfcal.py

Author: Jordan Mirocha
Affiliation: McGill
Created on: Sat 28 Mar 2020 14:59:01 EDT

Description: 

"""

import ares
import numpy as np
from ares.physics.Constants import rhodot_cgs

def test():
    
    mags = np.arange(-25, -5, 0.1)
    zarr = np.arange(6, 30, 0.1)
    
    pars = ares.util.ParameterBundle('mirocha2017:base')
    # Test suite only carries solar metallicity, heavily degraded, BPASS SEDs
    pars['pop_sed_degrade{0}'] = 100
    pars['pop_Z{0}'] = 0.02

    sim = ares.simulations.Global21cm(**pars)
    sim.run()
    
    sfrd = sim.pops[0].SFRD(zarr) * rhodot_cgs
    
    # Check for reasonable values
    assert np.all(sfrd < 1)
    assert 1e-6 <= np.mean(sfrd) <= 1e-1
    
    phi_M = sim.pops[0].LuminosityFunction(zarr[0], mags, mags=True, wave=1600.)
    
    assert 90 <= sim.nu_C <= 115, "Global signal unreasonable!"
    assert -250 <= sim.dTb_C <= -150, "Global signal unreasonable!"
    
if __name__ == '__main__':
    test()

    
    