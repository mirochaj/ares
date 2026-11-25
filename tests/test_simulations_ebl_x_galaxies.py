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
    zbins = np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]])
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
    # Check shape
    assert np.all(ptot.shape == (len(ell_modes), len(channels), len(zbins)))

    ##
    # Get the crossword
    crossword = ares.util.Misc.get_hmod_elements(sim, fluctuation_type=2)

    # Check that correct elements of inter-pop matrix are > 0
    

    
    ##
    # Could check that we can recover ptot from Limber integrals

    ##
    # Check that we get the same answer by summing result
    # in two bins (n-weighte) as we do doing one z bin in 
    # one shot

    # (pops, waves, zbins, 2) : last dim is (volume dens, surface dens)
    n = sim.num_by_pop[:,0,:,1].sum(axis=0)

    ptot_1z_a = np.sum(n * ptot[0,0,:]) / np.sum(n)

    ptot_1z_b = sim.get_ebl_x_galaxies(ell_modes,
        waves=channels, 
        zbins=np.array([[zbins.min(), zbins.max()]]),
        selection_criteria=selection, 
        masking_criteria=mask,
        cache_ipop_mtx=None,
        selection_symmetric=1, 
        masking_symmetric=1,
        **kwargs)
    
    print(ptot_1z_a, ptot_1z_b[0,0,0])
    
    assert np.allclose(ptot_1z_a, ptot_1z_b[0,0,0], rtol=1e-3), \
        "Fails reconstruction of C_ell in broad bin from sum of narrow bins!"
    

if __name__ == '__main__':
    test()