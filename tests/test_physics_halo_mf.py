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

    b6 = pop.halos.get_bias(6.)
    b8 = pop.halos.get_bias(8.)
    assert b8[iM] > b6[iM], "Bias should increase with redshift!"
    assert b6[iM+1] > b6[iM], "Bias should increase with mass!"

    # Test caching (motivated by PR #24)
    cache = pop.halos.tab_z, pop.halos.tab_M, pop.halos.tab_dndm

    pop2 = ares.populations.HaloPopulation(hmf_cache=cache)
    fcoll8_2 = pop2.halos.tab_fcoll[iz,iM]

    assert abs(fcoll8 - fcoll8_2) < 1e-8, \
        "Error in fcoll auto-generation: {:.12f} {:.12f}".format(fcoll8, fcoll8_2)

    # Test against HMF we generate now with hmf package.
    # Use narrow redshift range to speed-up, but keep redshift sampling high
    # to test MAR machinery.
    pop3 = ares.populations.HaloPopulation(hmf_load=False,
        hmf_zmin=6, hmf_zmax=7)

    iz3 = np.argmin(np.abs(6 - pop3.halos.tab_z))
    iM3 = np.argmin(np.abs(1e8 - pop3.halos.tab_M))

    dndm3 = pop3.halos.tab_dndm[iz3,:]

    fcoll8_3 = pop3.halos.tab_fcoll[iz3,iM3]

    # Compare real-time-generated HMF to tabulated HMF (pulled from Dropbox).
    assert pop.halos.tab_z[iz] == pop3.halos.tab_z[iz3]

    assert abs(fcoll8 - fcoll8_3) < 1e-2, \
        "Percent-level differences in tabulated and generated fcoll: {:.12f} {:.12f}".format(fcoll8, fcoll8_3)

    pop3.halos.save(clobber=True, save_MAR=True)
    #pop3.halos.save(clobber=True, save_MAR=True, fmt='pkl')

    assert np.allclose(dndm, dndm3, rtol=1e-2), \
        "Percent-level differences in tabulated and generated HMF!"


if __name__ == '__main__':
    test()
