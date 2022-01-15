"""

test_halo_model.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Fri Jul  8 12:24:24 PDT 2016

Description:

"""

import ares
import numpy as np

def test(rtol=1e-3):

    hm = ares.physics.HaloModel(hps_assume_linear=True, use_mcfit=True)
    hm_CC = ares.physics.HaloModel(hps_assume_linear=True, use_mcfit=False)

    # Just check large k range, k < 1 / Mpc
    k = np.logspace(-2, 0, 10)

    # Compare linear matter power spectrum to 2-h term of halo model
    for z in [20, 10, 6]:

        iz = np.argmin(np.abs(hm.tab_z - z))
        plin = np.exp(np.interp(np.log(k), np.log(hm.tab_k_lin),
            np.log(hm.tab_ps_lin[iz,:])))

        # Will default to using k-values used for plin if nothing supplied
        ps2h = hm.get_ps_2h(z, k)

        rerr = np.abs((plin - ps2h) / plin)
        assert np.allclose(plin, ps2h, rtol=rtol), \
            "2h term != linear matter PS at z={}. err_rel(k)={}".format(z,
                rerr)

    # Test CF w/ z=6
    R = np.logspace(-2, 3, 200)
    R, cf = hm.get_cf_mm(z, R=R)
    R2, cf2 = hm_CC.get_cf_mm(z, R)

    assert np.allclose(R, R2, rtol=1e-3), \
        "Disagreement between mcfit and Clenshaw-Curtis integrator for CF."


if __name__ == '__main__':
    test()
