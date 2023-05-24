"""

test_populations_cohort_funcs.py

Author: Jordan Mirocha
Affiliation: McGill
Created on: Wed 25 Mar 2020 09:35:24 EDT

Description:

"""

import ares
import numpy as np

def test():

    Mh = np.logspace(7, 15, 200)
    mag_bins = np.arange(-25, -5, 0.1)
    zarr = np.arange(6, 30, 0.1)

    # Initialize a simple DPL SFE model
    pars = ares.util.ParameterBundle('mirocha2017:base').pars_by_pop(0,1)
    pars.update(ares.util.ParameterBundle('testing:galaxies'))

    pop = ares.populations.GalaxyPopulation(**pars)

    # Check some basic attributes
    assert pop.is_synthesis_model
    assert pop.is_sfe_constant # in redshift!
    assert pop.is_metallicity_constant

    assert np.all(np.diff(pop.tab_Matom) < 0), "Mmin(z) should increase w/ z!"
    assert pop.get_Mmax(10) > pop.get_Mmin(10)

    assert np.all(pop.tab_focc == 1)

    sfrd = pop.get_sfrd(zarr)
    smd = pop.get_smd(zarr)

    # Check for reasonable values of SFRD, stellar mass density
    assert np.all(sfrd < 1)
    assert 1e-6 <= np.mean(sfrd) <= 1e-1

    # Check that phases of galaxy mass in right order
    Mhalo = pop.halos.tab_M
    Mst = pop.get_mass(10, 1e10, kind='stellar')
    Mg = pop.get_mass(10, 1e10, kind='gas')
    MZ = pop.get_mass(10, 1e10, kind='metals')

    assert 1e6 < Mst < 1e10
    assert Mst < Mg
    assert MZ < Mst

    # Really just a check of get_field method
    Mg_b = np.interp(1e10, pop.get_field(10, 'Mh'), pop.get_field(10, field='Mg'))
    assert abs(Mg - Mg_b) / Mg < 1e-4

    _Mh, zeta = pop.get_zeta(6.)
    assert 1 <= zeta.mean() <= 100, "zeta unreasonable!"

    L = pop.get_lum(6)
    mag = pop.get_mags(6, absolute=True)

    assert L.size == Mhalo.size
    assert mag.size == Mhalo.size

    Mh_17 = np.interp(-17, mag[-1::-1], Mhalo[-1::-1])

    sfrd_6_all = pop.get_sfrd(6.)
    sfrd_6_17 = pop.get_sfrd_in_mag_range(6, hi=-17, absolute=True)

    assert sfrd_6_17 < sfrd_6_all

    # Halo abundances
    assert pop.get_nh_active(6) > pop.get_nh_active(10)

    # Luminosity function and stellar mass functions
    x, phi_M = pop.get_lf(zarr[0], mag_bins, use_mags=True, x=1600., units='Angstroms')

    # A bit slow :/
    phi_Ms = pop.get_smf(zarr[0])

    mags, rho_surf = pop.get_surface_density(6.)

    assert -15 <= pop.get_mag_lim(6.) <= 0., \
        f"Limiting magnitude MUV={pop.get_mag_lim(6.)} unreasonable."

    Mmin = pop.get_Mmin(10.)

    assert 1e7 <= Mmin <= 1e9

    ##
    # Test metallicity stuff
    pars_Z = pars.copy()
    pars_Z['pop_enrichment'] = 1
    pars_Z['pop_metal_yield'] = 0.05
    pars_Z['pop_fpoll'] = 0.03
    pars_Z['pop_calib_lum'] = None
    pars_Z['pop_enrichment'] = True
    pop_Z = ares.populations.GalaxyPopulation(**pars_Z)

    Z = pop_Z.get_metallicity(6)

    # Don't apply 1e-3 floor anymore.
    #assert 1e-3 <= np.mean(Z) <= 0.04, \
    #    f"Mean metallicity not in tabulated range! Z={Z}"

    # Can't do this unless we download multiple BPASS tables when running
    # test suite.
    #x2, phi_M2 = pop_Z.get_lf(zarr[0], mag_bins, use_mags=True, wave=1600.)
#
    ## Should be different, but not crazy different
    #assert not np.all(phi_M == phi_M2)
    #ok = np.logical_and(x2 >= -24, x2 <= -15)
    #err = abs(phi_M[ok==1] - phi_M2[ok==1]) / phi_M[ok==1]
    #assert err.mean() < 0.5



if __name__ == '__main__':
    test()
