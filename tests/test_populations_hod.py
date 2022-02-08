
"""
test_populations_hod.py
Author: Emma Klemets
Affiliation: McGill
Created on: Aug 7, 2020

Description: Test the main functions of GalaxyHOD.py.
"""

import ares
import numpy as np

def test():
    #set up basic pop
    pars = ares.util.ParameterBundle('emma:model2')
    pop = ares.populations.GalaxyPopulation(**pars)

    z = 5
    mags = np.linspace(-24, -12)

    #test LF for high Z
    x, LF = pop.get_lf(z, mags)
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
    Zs = np.linspace(0, 6, 50)
    SFRD = pop.SFRD(Zs)
    assert all(1e-6 <= i <= 1 for i in SFRD), "SFRD unreasonable"

if __name__ == '__main__':
    test()
