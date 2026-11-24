"""

test_simulations_ebl_x_galaxies.py

Author: Jordan Mirocha
Affiliation: Caltech
Created on: Fri Nov 20 12:59:30 2026

Description:

"""

import ares
import pytest
import numpy as np

pytestmark = pytest.mark.slow

def test():

    pars = ares.util.ParameterBundle('mirocha2025:smhm_univ')
    pars['progress_bar'] = False
    sim = ares.simulations.Simulation(**pars)

    kwargs = {}
    ell_modes = np.array([1e4])
    channels = np.array([[1, 1.05], [2., 2.05]])
    zbins = np.array([[0.1, 0.2], [0.5, 0.6], [1., 1.1]])
    selection = {'mag': [('sdss_z', 22)]}
    mask = {'mag': [('sdss_z', 16)]}

    ptot = sim.get_ebl_x_galaxies(ell_modes,
        waves=channels, 
        zbins=zbins,
        selection_criteria=selection, 
        masking_criteria=mask,
        cache_ipop_mtx=None,
        selection_symmetric=1, 
        masking_symmetric=1,
        **kwargs)
    
    ##
    # Check the crossword

    
    ##
    # Could check that we can recover ptot from Limber integrals

    ##
    # How to check that the normalization is OK?

if __name__ == '__main__':
    test()