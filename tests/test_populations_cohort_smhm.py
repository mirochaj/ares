"""

test_populations_cohort_smhm.py

Author: Jordan Mirocha
Affiliation: JPL / Caltech
Created on: Sat Apr  8 12:09:15 PDT 2023

Description:

"""

import ares
import numpy as np

def test():

    Mh = np.logspace(7, 15, 200)
    mags = np.arange(-25, -10)
    waves = np.arange(900, 5000, 100)

    # Low resolution SEDs, HMF tables
    testing_pars = ares.util.ParameterBundle('testing:galaxies')

    parsSF = ares.util.ParameterBundle('mirocha2023:centrals_sf')
    parsSF.update(testing_pars)
    pop = ares.populations.GalaxyPopulation(**parsSF)

    x, phi = pop.get_lf(6, mags)

    focc = pop.get_focc(6, Mh)
    fsurv = pop.get_fsurv(6, Mh)

    assert not np.all(focc == 1)
    assert np.all(fsurv == 1)

    sfr = pop.get_sfr(6, Mh)
    ssfr = pop.get_ssfr(6, Mh)

    pars = ares.util.ParameterBundle('mirocha2023:base')
    testing_pars.num = 0
    pars.update(testing_pars)
    pars['pop_enrichment{0}'] = 0
    testing_pars.num = 1
    pars.update(testing_pars)
    pars['pop_enrichment{1}'] = 0
    sim = ares.simulations.Simulation(**pars)

    sim.pops[1].get_smd(6)
    assert np.all(sim.pops[2].get_focc(6, Mh) == 1. - sim.pops[0].get_focc(6, Mh))

    # Check luminosity, SEDs etc?
    L0 = sim.pops[0].get_lum(6, x=1600)
    L1 = sim.pops[1].get_lum(6, x=1600)

    Z = sim.pops[0].get_metallicity(6)
    assert np.all(Z == 0), \
        f"Metallicities should be zero! Mean is Z={np.mean(Z)}."

    metals = ares.util.ParameterBundle('mirocha2023:metals')
    metals.num = 0

    pars.update(metals)
    pars['pop_enrichment{0}'] = 1

    sim = ares.simulations.Simulation(**pars)

    Z = sim.pops[0].get_metallicity(6)
    #assert np.all(np.logical_and(1e-3 <= Z, Z < 1)), \
    #    f"Metallicities should be zero! Mean is Z={np.mean(Z)}."

    spec0 = sim.pops[0].get_spec(2, waves)
    spec1 = sim.pops[1].get_spec(2, waves)

    # Assert that the quiescent population is fainter at rest-UV wavelengths
    # than the star-forming population at fixed stellar mass
    Mst0 = sim.pops[0].get_field(2, 'Ms')
    Mst1 = sim.pops[1].get_field(2, 'Ms')

    i10_0 = np.argmin(np.abs(1e10 - Mst0))
    i10_1 = np.argmin(np.abs(1e10 - Mst1))

    assert np.all(spec0[i10_0,waves < 2000] > spec1[i10_1,waves < 2000])

    # Do the same thing with the emissivity
    assert sim.pops[0].get_emissivity(6, x=6, units='eV') \
         > sim.pops[1].get_emissivity(6, x=6, units='eV')


    dust = ares.util.ParameterBundle('mirocha2023:dust')
    dust.num = 0

    parsD = pars.copy()
    parsD.update(dust)
    parsD.update(testing_pars)
    simD = ares.simulations.Simulation(**parsD)

    # Make sure dust is reddening as it should.
    tau = simD.pops[0].get_dust_opacity(6, Mh, wave=5e3)
    assert np.any(tau > 0)
    assert np.all(simD.pops[0].get_spec(2, waves) <= spec0)




if __name__ == '__main__':
    test()
