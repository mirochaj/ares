  
"""
test_populations_hod.py
Author: Emma Klemets
Affiliation: McGill
Created on: Aug 7, 2020

Description: 
"""

import ares
import numpy as np

def test():
    pars = ares.util.ParameterBundle('emma:model2')

    pop = ares.populations.GalaxyPopulation(**pars)
    
#    assert 1e-3 <= sfrd <= 1, "SFRD unreasonable"
#    assert 1e4 <= np.mean(Md) <= 1e10, "Dust masses unreasonable!"
#     assert 1e-4 <= np.interp(-18, mags, phi) <= 1e-1, "UVLF unreasonable!"
#    assert np.array_equal(phi[ok==1], phi_c[ok==1]), "UVLF cache not working!"

    z = 5
    mags = np.linspace(-24, -12)
    
    #test LF for high Z
    LF = pop.LuminosityFunction(z, mags)
    assert all(1e-8 <= i <= 10 for i in LF), "LF unreasonable"
    
    log_HM = 0
    SM = pop.SMHM(2, log_HM)
         
    #test SMF
    z = 1.75
    logbins = np.linspace(7, 12, 60)
    
    SMF_tot = pop.StellarMassFunction(z, logbins)
    
    assert all(1e-19 <= i <= 1 for i in SMF_tot), "SMF unreasonable"
    
    SMF_sf = pop.StellarMassFunction(z, logbins, sf_type='smf_sf')
    SMF_q = pop.StellarMassFunction(z, logbins, sf_type='smf_q')
    
    assert all(np.less(SMF_sf, SMF_tot)), "Sf-fraction of SMF bigger than total"
    assert all(np.less(SMF_q, SMF_tot)), "Q-fraction of SMF bigger than total"
    
    SM = np.linspace(8, 11.1)
    #test SFR
    SFR = pop.SFR(z, SM)
    assert all(-2 <= i <= 3 for i in SFR), "SFR unreasonable"
    
    #test SSFR
    SSFR = pop.SSFR(z, SM)
    assert all(-10 <= i <= -7 for i in SSFR), "SSFR unreasonable"
    
    #test SFRD
    Zs = np.linspace(0, 8, 50)
    SFRD = pop.SFRD(Zs)
    assert all(1e-6 <= i <= 1 for i in SFRD), "SFRD unreasonable"

if __name__ == '__main__':
    test()

