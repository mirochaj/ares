"""

test_populations_popIII.py

Author: Jordan Mirocha
Affiliation: McGill
Created on: Tue  7 Apr 2020 21:18:13 EDT

Description:

"""

import ares
import numpy as np
from ares.physics.Constants import rhodot_cgs

def test():

    mags = np.arange(-25, -5, 0.1)
    zarr = np.arange(6, 30, 0.1)

    pars = ares.util.ParameterBundle('mirocha2017:base') \
         + ares.util.ParameterBundle('mirocha2018:high')

    updates = ares.util.ParameterBundle('testing:galaxies')
    updates.num = 0
    pars.update(updates)

    # Just testing: speed this up.
    pars['feedback_LW'] = False
    pars['feedback_LW_maxiter'] = 2
    pars['tau_redshift_bins'] = 400
    pars['hmf_dt'] = 1
    pars['hmf_tmax'] = 1000

    # Use sam_dz?

    sim = ares.simulations.Global21cm(**pars)
    sim.run()

    sfrd_II = sim.pops[0].SFRD(zarr) * rhodot_cgs
    sfrd_III = sim.pops[2].SFRD(zarr) * rhodot_cgs
    # Check for reasonable values
    assert np.all(sfrd_II < 1)
    assert 1e-6 <= np.mean(sfrd_II) <= 1e-1

    assert np.all(sfrd_III < 1)
    assert 1e-8 <= np.mean(sfrd_III) <= 1e-3

    phi_M = sim.pops[0].LuminosityFunction(zarr[0], mags, mags=True, wave=1600.)

    assert 60 <= sim.nu_C <= 115, "Global signal unreasonable!"
    assert -250 <= sim.dTb_C <= -150, "Global signal unreasonable!"


if __name__ == '__main__':
    test()
