"""

test_populations_cohort_funcs.py

Author: Jordan Mirocha
Affiliation: McGill
Created on: Wed 25 Mar 2020 09:35:24 EDT

Description:

"""

import ares
import numpy as np
from ares.physics.Constants import rho_cgs, rhodot_cgs

def test():

    Mh = np.logspace(7, 15, 200)
    mags = np.arange(-25, -5, 0.1)
    zarr = np.arange(6, 30, 0.1)

    # Initialize a simple DPL SFE model
    pars = ares.util.ParameterBundle('mirocha2017:base').pars_by_pop(0,1)
    pars.update(ares.util.ParameterBundle('testing:galaxies'))

    pop = ares.populations.GalaxyPopulation(**pars)

    # Check some basic attributes
    assert pop.is_synthesis_model
    assert not pop.is_sfr_constant
    assert pop.is_sfe_constant # in redshift!

    assert np.all(pop.tab_focc == 1)

    sfrd = pop.get_sfrd(zarr) * rhodot_cgs
    smd = pop.get_smd(zarr) * rho_cgs

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
    x, phi_M = pop.get_lf(zarr[0], mags, use_mags=True, wave=1600.)

    # A bit slow :/
    phi_Ms = pop.get_smf(zarr[0])

    mags, rho_surf = pop.get_surface_density(6.)

    dsfe_dMh = pop.get_sfe_slope(6., 1e9)

    assert abs(dsfe_dMh - pop.pf['pq_func_par2[0]']) < 0.05

    assert -15 <= pop.get_MUV_lim(6.) <= 0., "Limiting magnitude unreasonable."

    Mmin = pop.get_Mmin(10.)

    assert 1e7 <= Mmin <= 1e9


if __name__ == '__main__':
    test()
