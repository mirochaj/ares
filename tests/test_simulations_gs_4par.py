"""

test_gs_basic.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Thu Jan 19 10:35:33 PST 2017

Description:

"""

import ares
import numpy as np

def test():
    sim = ares.simulations.Global21cm()

    sim.info
    pf = sim.pf
    sim.pf._check_for_conflicts()
    assert sim.pf.Npops == 3

    sim.run()

    #
    # Make sure it's not a null signal.
    z = sim.history['z']
    dTb = sim.history['dTb'][z < 50]
    assert len(np.unique(np.sign(dTb))) == 2
    assert max(dTb) > 5 and min(dTb) < -5

    # Test that the turning points are there, that tau_e is reasonable, etc.
    assert 80 <= sim.z_A <= 90
    assert 10 <= sim.nu_A <= 20
    assert -50 <= sim.dTb_A <= -40

    assert 25 <= sim.z_B <= 35
    assert -15 <= sim.dTb_B <= 0

    assert 10 <= sim.z_C <= 25
    assert -250 <= sim.dTb_C <= 0

    assert 6 <= sim.z_D <= 15
    assert 0 <= sim.dTb_D <= 30

    assert 0.04 <= sim.tau_e <= 0.15

    fwhm = sim.Width()
    hwhm = sim.Width(peak_relative=True)

    assert 10 <= fwhm <= 50
    assert 0 <= hwhm <= 3

    k = sim.kurtosis
    s = sim.skewness

    slope1 = sim.dTbdz
    slope2 = sim.dTbdnu
    curv1 = sim.dTb2dz2
    curv2 = sim.dTb2dnu2


    sim.save('test_gs_4par', 'pkl', clobber=True)
    sim.save('test_gs_4par', 'txt', clobber=True)

if __name__ == '__main__':
    test()
