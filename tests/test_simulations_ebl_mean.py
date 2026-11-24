"""

test_simulations_ebl_mean.py

Author: Jordan Mirocha
Affiliation: Caltech
Created on: Mon Nov 23 12:01:03 2026

Description:

"""

import ares
import pytest
import numpy as np

pytestmark = pytest.mark.slow

# Will be slow, need to first try to figure out speed-up
def test():
    
    pars = ares.util.ParameterBundle('mirocha2025:smhm_univ')
    pars['progress_bar'] = False
    pars['tau_redshift_bins'] = 50
    pars['initial_redshift'] = 2
    pars['first_light_redshift'] = 2

    sim = ares.simulations.Simulation(**pars)

    dlam = 0.5
    #bands = np.array([(lo, lo+dlam) \
    #    for lo in np.arange(0.5, 5.+dlam, dlam)])    

    ##
    # For speed, let's just do this in one band.
    bands = np.array([[0.5, 0.6]])

    # First, let's compute the mean flux at z=0 via
    # integration of the radiative transfer equation.
    ebl_via_rte = sim.get_ebl_intensity(
        compute_via_counts=0, bands=bands
    )
    
    ##
    # Let's compare to the mean EBL we get from integrating
    # galaxy counts in the same bands.
    magbins = np.arange(10, 41, 1.)

    ebl_via_cts = sim.get_ebl_intensity(
        compute_via_counts=1, bands=bands, magbins=magbins,
        zmin=0, zmax=pars['first_light_redshift'],
    )

    ##
    # Check for consistency. Can do population by population.
    for i in ebl_via_rte.keys():
        x_rte, y_rte = ebl_via_rte[i]
        x_cts, y_cts = ebl_via_cts[i]
        # This should pass with flying colors: just provided bands
        assert np.allclose(x_rte, x_cts, rtol=1e-8), \
            f"Failed to recover same bands for pop={i}!"
        # This is less clear. Should be close but not exact.
        # Let's start with a 10% tolerance and see how things go.
        assert np.allclose(y_rte, y_cts, rtol=1e-1), \
            f"Failed to recover same mean EBL (cts v. RTE) for pop={i}"


if __name__ == '__main__':
    test()

    