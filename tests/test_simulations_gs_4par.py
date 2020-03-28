"""

test_gs_basic.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Thu Jan 19 10:35:33 PST 2017

Description: 

"""

import ares
import numpy as np
import matplotlib.pyplot as pl

def test():
    sim = ares.simulations.Global21cm()
    sim.run()
    ax, zax = sim.GlobalSignature(fig=1, ymin=-400)
    
    sim.AdiabaticFloor(ax, color='k', ls=':')
    sim.AdiabaticFloor(ax)
    
    sim.SaturatedLimit(ax)
    
    inset_tau = sim.add_tau_inset(ax)
    inset_Ts = sim.add_Ts_inset(ax)
    
    #
    # Make sure it's not a null signal.
    z = sim.history['z']
    dTb = sim.history['dTb'][z < 50]
    assert len(np.unique(np.sign(dTb))) == 2
    assert max(dTb) > 5 and min(dTb) < -5
    
    # Test that the turning points are there, that tau_e is reasonable, etc.
    assert 80 <= sim.z_A <= 90
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
    
    ax2 = sim.OpticalDepthHistory(fig=2)
    ax3 = sim.TemperatureHistory(fig=3)
    ax4 = sim.GlobalSignatureDerivative(fig=4)
    ax5 = sim.GlobalSignatureDerivative(fig=5, show_signal=True)
    
    for i in range(1, 6):
        pl.figure(i)
        pl.savefig('{0!s}_{1}.png'.format(__file__[0:__file__.rfind('.')], i))     
    
    pl.close('all')
    
if __name__ == '__main__':
    test()


