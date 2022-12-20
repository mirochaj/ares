"""

test_physics_nebula.py

Author: Jordan Mirocha
Affiliation: McGill
Created on: Sun  7 Jun 2020 16:31:42 EDT

Description:

"""

import sys
import ares
import numpy as np
from ares.physics.Constants import h_p, c, erg_per_ev, lam_LyA, E_LL

def test():

    # Setup pure continuum source
    pars_con = ares.util.ParameterBundle('mirocha2017:base').pars_by_pop(0, 1)
    pars_con.update(ares.util.ParameterBundle('testing:galaxies'))
    pars_con['pop_nebular'] = 0
    pop_con = ares.populations.GalaxyPopulation(**pars_con)

    # Source with nebular continuum and H-lines and built-in
    # treatment for continuum coefficients
    pars_ares = ares.util.ParameterBundle('mirocha2017:base').pars_by_pop(0, 1)
    pars_ares.update(ares.util.ParameterBundle('testing:galaxies'))
    pars_ares['pop_nebular'] = 2
    pop_ares = ares.populations.GalaxyPopulation(**pars_ares)

    # Nebular emission w/ coefficients from Ferland (1980)
    pars_ares2 = ares.util.ParameterBundle('mirocha2017:base').pars_by_pop(0, 1)
    pars_ares2.update(ares.util.ParameterBundle('testing:galaxies'))
    pars_ares2['pop_nebular'] = 2
    pars_ares2['pop_nebular_lookup'] = 'ferland1980'
    pop_ares2 = ares.populations.GalaxyPopulation(**pars_ares2)

    # Setup source with BPASS-generated (CLOUDY) nebular emission
    pars_sps = ares.util.ParameterBundle('mirocha2017:base').pars_by_pop(0, 1)
    pars_sps.update(ares.util.ParameterBundle('testing:galaxies'))
    pars_sps['pop_nebular'] = 1
    pars_sps['pop_fesc'] = 0.
    pars_sps['pop_nebular_Tgas'] = 2e4
    pop_sps = ares.populations.GalaxyPopulation(**pars_sps)

    for k, t in enumerate([1, 5, 10, 20, 50]):
        i = np.argmin(np.abs(pop_ares.src.tab_t - t))

        # For some reason, the BPASS+CLOUDY tables only go up to 29999A,
        # so the degraded tables will be one element shorter than their
        # pop_nebular=False counterparts. So, interpolate for errors.
        # (this is really just making shapes the same, since common
        # wavelengths will be identical)
        y_ares = np.interp(pop_sps.src.tab_waves_c,
            pop_ares.src.tab_waves_c, pop_ares.src.tab_sed[:,i])
        y_ares2 = np.interp(pop_sps.src.tab_waves_c,
            pop_ares2.src.tab_waves_c, pop_ares2.src.tab_sed[:,i])
        err = np.abs(y_ares - pop_sps.src.tab_sed[:,i]) / pop_sps.src.tab_sed[:,i]
        err2 = np.abs(y_ares2 - pop_sps.src.tab_sed[:,i]) / pop_sps.src.tab_sed[:,i]

        Lion_H = pop_ares.src._nebula.get_ion_lum(pop_ares.src.tab_sed[:,i], 0)
        Lion_He = pop_ares.src._nebula.get_ion_lum(pop_ares.src.tab_sed[:,i], 1)
        Lion_He2 = pop_ares.src._nebula.get_ion_lum(pop_ares.src.tab_sed[:,i], 2)

        assert Lion_H > Lion_He
        assert Lion_He > Lion_He2

        Nion_H = pop_ares.src._nebula.get_ion_num(pop_ares.src.tab_sed[:,i], 0)
        Nion_He = pop_ares.src._nebula.get_ion_num(pop_ares.src.tab_sed[:,i], 1)
        Nion_He2 = pop_ares.src._nebula.get_ion_num(pop_ares.src.tab_sed[:,i], 2)

        assert Nion_H > Nion_He
        assert Nion_He > Nion_He2

        Eion_H = pop_ares.src._nebula.get_ion_Eavg(pop_ares.src.tab_sed[:,i], 0)
        Eion_He = pop_ares.src._nebula.get_ion_Eavg(pop_ares.src.tab_sed[:,i], 1)
        Eion_He2 = pop_ares.src._nebula.get_ion_Eavg(pop_ares.src.tab_sed[:,i], 2)

        assert E_LL < Eion_H < 24.6
        assert 24.6 < Eion_He < 4 * E_LL
        assert Eion_He2 > E_LL * 4


    # Make sure emission at Ly-a is brighter for population with nebular
    # lines included.
    ilya = np.argmin(np.abs(pop_ares.src.tab_waves_c - lam_LyA))
    assert np.all(pop_ares.src.tab_sed[ilya,:] > pop_con.src.tab_sed[ilya,:])

    # Make sure emission at H-a is brighter for population with nebular
    # lines included.
    EHa = pop_ares.src._nebula.hydr.BohrModel(ninto=2, nfrom=3)
    iHa = np.argmin(np.abs(pop_ares.src.tab_energies_c - EHa))
    assert np.all(pop_ares.src.tab_sed[iHa,:] > pop_con.src.tab_sed[iHa,:])

    # Make sure emission in rest-UV continuum is brighter for population with
    # nebular continuum+lines included.
    i1400 = np.argmin(np.abs(pop_ares.src.tab_waves_c - 1400))
    assert np.all(pop_ares.src.tab_sed[i1400,:] > pop_con.src.tab_sed[i1400,:])

    # Check the Ferland (1980) vs default (Dopita & Sutherland) treatments
    i1000 = np.argmin(np.abs(pop_ares.src.tab_waves_c - 1000.))
    err = abs(pop_ares.src.tab_sed[i1000,:] - pop_ares2.src.tab_sed[i1000,:]) \
        / pop_ares.src.tab_sed[i1000,:]
    assert np.all(err <= 1e-2), \
        "Ferland (1980) results should be closer to Dopita \& Sutherland!"


if __name__ == '__main__':
    test()
