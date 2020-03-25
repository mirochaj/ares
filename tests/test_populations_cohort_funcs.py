"""

test_populations_cohort_funcs.py

Author: Jordan Mirocha
Affiliation: McGill
Created on: Wed 25 Mar 2020 09:35:24 EDT

Description: 

"""

import ares
import numpy as np
import matplotlib.pyplot as pl
from ares.physics.Constants import rhodot_cgs

def test():
    
    Mh = np.logspace(7, 15, 200)
    mags = np.arange(-25, -5, 0.1)
    zarr = np.arange(6, 30, 0.1)
    
    pars = ares.util.ParameterBundle('mirocha2017:base').pars_by_pop(0,1)
    pars['pop_L1600_per_sfr'] = 1e28
    pop = ares.populations.GalaxyPopulation(**pars)
    
    sfrd = pop.SFRD(zarr) * rhodot_cgs
    
    # Check for reasonable values
    assert np.all(sfrd < 1)
    assert 1e-6 <= np.mean(sfrd) <= 1e-1
    
    # UVLF: need a population synthesis model.
    phi_M = pop.LuminosityFunction(zarr[0], mags, mags=True, wave=1600.)
    
if __name__ == '__main__':
    test()

    
    