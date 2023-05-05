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
    pars = ares.util.ParameterBundle('global_signal:basic')
    sim = ares.simulations.Simulation(**pars)

    sim.sim_gs.info
    pf = sim.pf
    sim.sim_gs.pf._check_for_conflicts()
    assert sim.pf.Npops == 3

    sim_gs = sim.get_21cm_gs()

    #
    # Make sure it's not a null signal.
    z = sim_gs.history['z']
    dTb = sim_gs.history['dTb'][z < 50]
    assert len(np.unique(np.sign(dTb))) == 2
    assert max(dTb) > 5 and min(dTb) < -5

    # Test that the turning points are there, that tau_e is reasonable, etc.
    assert 80 <= sim_gs.z_A <= 90
    assert 10 <= sim_gs.nu_A <= 20
    assert -50 <= sim_gs.dTb_A <= -40

    assert 25 <= sim_gs.z_B <= 35
    assert -15 <= sim_gs.dTb_B <= 0

    assert 10 <= sim_gs.z_C <= 25
    assert -250 <= sim_gs.dTb_C <= 0

    assert 6 <= sim_gs.z_D <= 15
    assert 0 <= sim_gs.dTb_D <= 30

    assert 0.04 <= sim_gs.tau_e <= 0.15

    fwhm = sim_gs.Width()
    hwhm = sim_gs.Width(peak_relative=True)

    assert 10 <= fwhm <= 50
    assert 0 <= hwhm <= 3

    k = sim_gs.kurtosis
    s = sim_gs.skewness

    slope1 = sim_gs.dTbdz
    slope2 = sim_gs.dTbdnu
    curv1 = sim_gs.dTb2dz2
    curv2 = sim_gs.dTb2dnu2

    # Save, read back in
    sim_gs.save('test', suffix='pkl', clobber=True)
    sim_gs.save('test', suffix='hdf5', clobber=True)

    sim_gs2 = ares.analysis.Global21cm('test')
    assert np.all(sim_gs.history['cgm_h_2'] == sim_gs2.history['cgm_h_2'])

if __name__ == '__main__':
    test()
