"""

test_physics_HI_tau.py

Author: Jordan Mirocha
Affiliation: McGill University
Created on: Fri 11 Feb 2022 12:06:01 EST

Description:

"""

import numpy as np
from ares.physics import Hydrogen

def test(rtol=1e-1):
    hydr_approx = Hydrogen(approx_tau_21cm=True)
    hydr_general = Hydrogen(approx_tau_21cm=False)

    for i, z in enumerate([10, 20, 30]):
        dTb_a = hydr_approx.get_21cm_adiabatic_floor(z)
        dTb_g = hydr_general.get_21cm_adiabatic_floor(z)

        err_r = np.abs((dTb_a - dTb_g) / dTb_g)

        assert np.all(err_r < rtol)

if __name__ == '__main__':
    test()
