
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
    pars = ares.util.ParameterBundle('mirocha2020:legacy_irxb') \
         + ares.util.ParameterBundle('sun2020:_nirb_updates') \
         + ares.util.ParameterBundle('pop:hod') \
         + ares.util.ParameterBundle('testing:galaxies')

    pop = ares.populations.GalaxyPopulation(**pars)

    z = 6
    mags = np.linspace(-22, -12)

    # Make sure LF is reasonable
    x, phi = pop.get_lf(z, mags)
    assert np.all(np.logical_and(1e-8 <= phi, phi <= 10)), \
        f"LF unreasonable: {phi}"

    z = 2
    assert pop.get_smhm(z=z, Mh=1e10) < pop.get_smhm(z=z, Mh=1e12)

    #test SMF
    logbins = np.arange(7, 11, 0.2)

    bins, smf_tot = pop.get_smf(z, logbins)

    assert np.all(np.logical_and(1e-8 <= smf_tot, smf_tot <= 10)), \
        "SMF unreasonable"


    #test SFR
    assert pop.get_sfr(z=z, Mh=1e10) < pop.get_sfr(z=z, Mh=1e12)

    #test SSFR
    assert pop.get_ssfr(z, Ms=1e10) < pop.get_ssfr(z+1, Ms=1e10)

    #test SFRD
    SFRD = pop.get_sfrd(8)
    assert 1e-4 <= SFRD <= 1, f"SFRD={SFRD} unreasonable"

if __name__ == '__main__':
    test()
