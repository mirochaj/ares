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
from ares.physics.Constants import h_p, c, erg_per_ev, lam_LyA

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
    pop_ares2 = ares.populations.GalaxyPopulation(**pars_ares)

    # Setup source with BPASS-generated (CLOUDY) nebular emission
    pars_sps = ares.util.ParameterBundle('mirocha2017:base').pars_by_pop(0, 1)
    pars_sps.update(ares.util.ParameterBundle('testing:galaxies'))
    pars_sps['pop_nebular'] = 1
    pars_sps['pop_fesc'] = 0.
    pars_sps['pop_nebula_Tgas'] = 2e4
    pop_sps = ares.populations.GalaxyPopulation(**pars_sps)

    code = 'bpass' if pars_ares['pop_sed'] == 'eldridge2009' else 's99'

    colors = 'k', 'b', 'c', 'm', 'r'
    for k, t in enumerate([1, 5, 10, 20, 50]):
        i = np.argmin(np.abs(pop_ares.src.times - t))

        # For some reason, the BPASS+CLOUDY tables only go up to 29999A,
        # so the degraded tables will be one element shorter than their
        # pop_nebular=False counterparts. So, interpolate for errors.
        # (this is really just making shapes the same, since common
        # wavelengths will be identical)
        y_ares = np.interp(pop_sps.src.wavelengths,
            pop_ares.src.wavelengths, pop_ares.src.data[:,i])
        y_ares2 = np.interp(pop_sps.src.wavelengths,
            pop_ares2.src.wavelengths, pop_ares2.src.data[:,i])
        err = np.abs(y_ares - pop_sps.src.data[:,i]) / pop_sps.src.data[:,i]
        err2 = np.abs(y_ares2 - pop_sps.src.data[:,i]) / pop_sps.src.data[:,i]

    # Make sure emission at Ly-a is brighter for population with nebular
    # lines included.
    ilya = np.argmin(np.abs(pop_ares.src.wavelengths - lam_LyA))
    assert np.all(pop_ares.src.data[ilya,:] > pop_con.src.data[ilya,:])

    # Make sure emission at H-a is brighter for population with nebular
    # lines included.
    EHa = pop_ares.src._nebula.hydr.BohrModel(ninto=2, nfrom=3)
    iHa = np.argmin(np.abs(pop_ares.src.energies - EHa))
    assert np.all(pop_ares.src.data[iHa,:] > pop_con.src.data[iHa,:])

    # Make sure emission in rest-UV continuum is brighter for population with
    # nebular continuum+lines included.
    i1400 = np.argmin(np.abs(pop_ares.src.wavelengths - 1400))
    assert np.all(pop_ares.src.data[i1400,:] > pop_con.src.data[i1400,:])


if __name__ == '__main__':
    test()
