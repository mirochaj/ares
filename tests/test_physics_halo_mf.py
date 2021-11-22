"""

test_physics_hmf.py

Author: Jordan Mirocha
Affiliation: McGill
Created on: Wed 25 Mar 2020 11:15:35 EDT

Description:

"""

import ares
import numpy as np
import matplotlib.pyplot as pl

def test():
    pop = ares.populations.HaloPopulation()

    m = pop.halos.tab_M
    iz = np.argmin(np.abs(6 - pop.halos.tab_z))
    iM = np.argmin(np.abs(1e8 - pop.halos.tab_M))

    dndm = pop.halos.tab_dndm[iz,:]
    
    fcoll8 = pop.halos.tab_fcoll[iz,iM]

    # Test caching (motivated by PR #24)
    cache = pop.halos.tab_z, pop.halos.tab_M, pop.halos.tab_dndm

    pop2 = ares.populations.HaloPopulation(hmf_cache=cache)
    fcoll8_2 = pop2.halos.tab_fcoll[iz,iM]

    assert abs(fcoll8 - fcoll8_2) < 1e-8, \
        "Error in fcoll auto-generation: {:.12f} {:.12f}".format(fcoll8, fcoll8_2)

if __name__ == '__main__':
    test()
