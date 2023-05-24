"""

test_populations_cohort_emissivity.py

Author: Jordan Mirocha
Affiliation: McGill University
Created on: Fri 31 Dec 2021 16:43:46 EST

Description:

"""

import ares
import numpy as np
import matplotlib.pyplot as pl
from ares.physics.Constants import cm_per_mpc, E_LL

def test():

    # Default model, constant f_esc, Nion
    pars = ares.util.ParameterBundle('mirocha2017:base')
    updates_uv = ares.util.ParameterBundle('testing:galaxies')
    updates_uv.num = 0
    pars.update(updates_uv)

    # Test that Mh-dep fesc, Nion, etc. work and lead to qualitative
    # shifts we expect.

    sim = ares.simulations.Simulation(**pars)
    pop_uv = sim.pops[0]
    pop_xr = sim.pops[1]

    assert pop_uv.is_emissivity_scalable
    assert pop_xr.is_emissivity_scalable

    zarr = np.arange(6, 30, 1)

    L_lw = np.array([pop_uv.get_emissivity(z, band=(10.2, 13.6)) \
        for z in zarr])
    L_ion = np.array([pop_uv.get_emissivity(z, band=(13.6, 1e2)) \
        for z in zarr])
    L_X = np.array([pop_xr.get_emissivity(z, band=(5e2, 8e3)) \
        for z in zarr])

    N_ion = np.array([pop_uv.get_photon_emissivity(z, band=(13.6, 1e2)) \
        for z in zarr])

    assert np.all(L_X < L_ion)
    assert np.all(L_ion < L_lw)
    assert 1e37 <= L_X[0] <= 1e39

    # Should check against solution obtained with pop_emissivity_tricks=False
    pars_no_trx = pars.copy()
    pars_no_trx['pop_emissivity_tricks{0}'] = False

    sim_no_trx = ares.simulations.Simulation(**pars_no_trx)
    pop_uv2 = sim_no_trx.pops[0]
    assert not pop_uv2.is_emissivity_scalable

    L_ion2  = np.array([pop_uv2.get_emissivity(z, band=(13.6, 1e2), units='eV') \
        for z in zarr])

    err_i = np.abs((L_ion - L_ion2) / L_ion)

    print(err_i)

    #import matplotlib.pyplot as pl
    #pl.semilogy(zarr, err_i, color='k')
    #pl.semilogy(zarr, err_x, color='b')
    #pl.ylim(1e-12, 1e2)
    #input('<enter>')
    assert np.all(L_ion2 == L_ion), f"err={err.mean()}"

    ##
    # Make fesc=fesc(Mh)
    pars_fesc = pars.pars_by_pop(0, 1)
    pars_fesc['pop_fesc'] = 'pq[1]'

    pars_fesc['pq_func[1]'] = 'pl'
    pars_fesc['pq_func_var[1]'] = 'Mh'
    pars_fesc['pq_func_par0[1]'] = 0.1
    pars_fesc['pq_func_par1[1]'] = 1e10
    pars_fesc['pq_func_par2[1]'] = -0.5

    pop_fesc = ares.populations.GalaxyPopulation(**pars_fesc)

    assert not pop_fesc.is_emissivity_scalable

    assert pop_fesc.get_fesc(z=None, Mh=1e10, x=13.62, units='ev') \
         < pop_fesc.get_fesc(z=None, Mh=1e9,  x=13.62, units='ev')

    L_ion2 = np.array([pop_fesc.get_emissivity(z, band=(E_LL, 24.6)) \
        for z in zarr])
    N_ion2 = np.array([pop_fesc.get_photon_emissivity(z, band=(E_LL, 24.6)) \
        for z in zarr])

    # If low-mass halos dominate, emissivity should evolve more gradually at late
    # times.
    d1 = -np.diff(L_ion) / np.diff(zarr)
    d2 = -np.diff(L_ion2) / np.diff(zarr)
    assert np.mean(d2) < np.mean(d1)

    d1 = -np.diff(N_ion) / np.diff(zarr)
    d2 = -np.diff(N_ion2) / np.diff(zarr)
    assert np.mean(d2) < np.mean(d1)

if __name__ == '__main__':
    test()
