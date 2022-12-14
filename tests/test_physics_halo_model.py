"""

test_halo_model.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Fri Jul  8 12:24:24 PDT 2016

Description:

"""

import numpy as np
from ares.physics import HaloModel

def test(rtol=1e-3):

    # Linear matter PS by default
    hm = HaloModel()

    # Just check large k range, k < 1 / Mpc
    k = np.logspace(-2, 0, 10)
    r = np.logspace(-5, 3)

    z = 6
    Mh = 1e10

    # Check NFW
    rho = hm.get_rho_nfw(z, Mh, r)
    rho_ft = hm.get_u_nfw(z, Mh, k)

    # Check Mh scaling very roughly
    assert hm.get_u_nfw(z, 1e10, 1e1) > hm.get_u_nfw(z, 1e12, 1e1)

if __name__ == '__main__':
    test()
