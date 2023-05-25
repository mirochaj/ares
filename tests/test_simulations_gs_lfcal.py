"""

test_simulations_gs_lfcal.py

Author: Jordan Mirocha
Affiliation: McGill
Created on: Sat 28 Mar 2020 14:59:01 EDT

Description:

"""

import ares
import numpy as np

def test():

    mags = np.arange(-25, -5, 0.1)
    zarr = np.arange(6, 30, 0.1)

    pars = ares.util.ParameterBundle('mirocha2017:base')
    # Test suite only carries solar metallicity, heavily degraded, BPASS SEDs
    pars['pop_sed_degrade{0}'] = 100
    pars['pop_Z{0}'] = 0.02

    sim = ares.simulations.Simulation(**pars)
    sim_gs = sim.get_21cm_gs()

    sfrd = sim.pops[0].get_sfrd(zarr)

    # Check for reasonable values
    assert np.all(sfrd < 1)
    assert 1e-6 <= np.mean(sfrd) <= 1e-1

    x, phi_M = sim.pops[0].get_lf(zarr[0], mags, use_mags=True,
        x=1600., units='Angstroms')

    assert 90 <= sim_gs.nu_C <= 115, \
        "Global signal unreasonable! nu_min={:.1f} MHz".format(sim_gs.nu_C)
    assert -250 <= sim_gs.dTb_C <= -150, \
        "Global signal unreasonable! dTb_min={:.1f} mK".format(sim_gs.dTb_C)

if __name__ == '__main__':
    test()
