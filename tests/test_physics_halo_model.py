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

    hm = HaloModel(hps_assume_linear=True)

    # Just check large k range, k < 1 / Mpc
    k = np.logspace(-2, 0, 10)
    r = np.logspace(-5, 3)

    z = 6
    Mh = 1e10

    # Check NFW
    rho = hm.get_profile(z, Mh, r, prof='nfw')
    rho_ft = hm.get_profile_FT(z, Mh, k, prof='nfw')

    # Check all inverse FT'd profiles
    for prof in hm.available_profiles:
        rho_ft = hm.get_profile_FT(z, Mh, k, prof=prof)


if __name__ == '__main__':
    test()
