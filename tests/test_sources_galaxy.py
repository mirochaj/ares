"""

test_sources_galaxy.py

Author: Jordan Mirocha
Affiliation: JPL / Caltech
Created on: Tue Nov 28 13:42:14 PST 2023

Description:

"""

import ares
import numpy as np


def test():
    testing_pars = ares.util.ParameterBundle('testing:galaxies')

    pars = {}
    pars['source_aging'] = True
    pars['source_ssp'] = True
    pars['source_sed_degrade'] = testing_pars['pop_sed_degrade']
    pars['source_sed'] = testing_pars['pop_sed']
    pars['source_Z'] = testing_pars['pop_Z']
    pars['source_ssp'] = True
    pars['source_sfh'] = 'exp_decl'
    pars['source_sfh_fallback'] = 'exp_rise'

    #pars_g['pop_enrichment'] = False
    galaxy = ares.sources.Galaxy(**pars)

    # Testing parameterized SFHs
    tobs = 1.3e4
    tarr = np.arange(100, 1.37e4, 10)
    sfr = 1
    mass = 1e12

    kw = galaxy.get_kwargs(tobs, mass, sfr, sfh='exp_decl',
        mass_return=False, tarr=tarr, mtol=0.05)
    sfh = galaxy.get_sfr(tarr, tobs, **kw)

    # Make sure the integral of the SFH = the mass we asked for
    m = np.trapz(sfh, x=tarr * 1e6)

    assert abs(m - mass) / mass < 0.05, \
        "Error in SFH! Recovered mass not accurate to 5%."

    waves = np.arange(900, 2e4, 100)
    spec = galaxy.get_spec(1, t=tarr, sfh=sfh, waves=waves)

if __name__ == '__main__':
    test()
