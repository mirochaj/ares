"""

test_physics_lya_coupling.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Sep 25 10:01:27 MDT 2013

Description:

"""

import ares
import numpy as np

def test():
    Tarr = np.logspace(-1, 2)

    res = []
    for i, method in enumerate([2,3,3.5,4,5]):
        hydr = ares.physics.Hydrogen(approx_Salpha=method)

        Sa = np.array([hydr.Sa(20., Tarr[k]) for k in range(Tarr.size)])
        res.append(Sa)

        if method == 5:
            # Mittal & Kulkarni (2018) quote a value in the text for
            # (z, x_e, Tk) = (22, 0, 10)
            assert abs(hydr.Sa(z=22., Tk=10.) - 0.7) < 1e-2

        # Check Ts while we're here
        Ts = hydr.get_Ts(20., hydr.cosm.Tgas(20.), 1, 0., 0.)

    # Compare at T > 1 K
    ok = Tarr > 1.
    diff = np.abs(np.diff(res, axis=1))

    # Just set to a level that I know is tight enough to pickup
    # any errors we might accidentally introduce later.
    assert np.all(diff.ravel() < 0.3)

    # Check frec
    for n in range(2, 31):
        frec = hydr.frec(n)

    assert hydr.Tbg is None

    # Check various limits
    dTb_sat = hydr.get_21cm_saturated_limit(10.)
    dTb_low = hydr.get_21cm_adiabatic_floor(10.)
    dTb_phy = hydr.get_21cm_dTb_no_astrophysics(10.)

    assert 0 <= dTb_sat <= 50
    assert -350 <= dTb_low <= -200
    assert abs(dTb_phy) < 1

if __name__ == '__main__':
    test()
