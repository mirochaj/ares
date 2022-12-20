"""

test_spec_synth.py

Author: Jordan Mirocha
Affiliation: McGill
Created on: Mon 20 May 2019 15:13:21 EDT

Description:

"""

import time
import ares
import numpy as np
from ares.physics.Constants import s_per_myr

def test(show_bpass=False, oversample_age=30., dt_coarse=10):

    toy = ares.sources.SynthesisModelToy(source_dlam=10., source_lmin=1e3,
        source_lmax=3e3, source_toysps_beta=-2, source_toysps_alpha=8.,
        source_ssp=True, source_aging=True)

    # Just checking
    E = toy.tab_energies_c
    dE = toy.dE
    dndE = toy.dndE
    f = toy.frequencies

    pars = ares.util.ParameterBundle('mirocha2020:univ')
    pars['pop_sed'] = 'sps-toy'
    pars['pop_toysps_beta'] = -2.
    # Turn off aging so we recover beta = -2
    pars["pop_toysps_alpha"] = 0.
    pars['pop_toysps_gamma'] = 0.
    pars['pop_dust_yield'] = 0
    pars['pop_lmin'] = 1000
    pars['pop_lmax'] = 3000
    pars['pop_dlam'] = 10.
    pars['pop_thin_hist'] = 0
    pars['pop_scatter_mar'] = 0
    pars['pop_Tmin'] = None # So we don't have to read in HMF table for Mmin
    pars['pop_Mmin'] = 1e8
    pars['pop_synth_minimal'] = False
    pars['tau_clumpy'] = None
    pars['pop_sed_degrade'] = None

    # Prevent use of hmf table
    tarr = np.arange(50, 1000, 1.)[-1::-1]
    zarr = toy.cosm.z_of_t(tarr * s_per_myr)
    pars['pop_histories'] = {'t': tarr, 'z': zarr,
        'MAR': np.ones((10, tarr.size)), 'nh': np.ones((10, tarr.size)),
        'Mh': 1e10 * np.ones((10, tarr.size))}

    pop1 = ares.populations.GalaxyPopulation(**pars)

    if show_bpass:
        src = ares.sources.SynthesisModel(source_sed='eldridge2009',
            source_ssp=True, source_Z=0.004)

    ##
    # Experiment with toy models.
    # Plot L(t) for three different wavelengths.
    # Plot full spectrum at a few different times
    ##
    tarr = np.logspace(0, 4, 100)

    for i, wave in enumerate([900., 1200., 1600., 2300.]):

        # Plot parameteric model solution
        #ax1.loglog(tarr, L(tarr, wave=wave), color=colors[i], ls='--')

        y2 = toy.tab_sed[np.argmin(np.abs(toy.tab_waves_c - wave)),:]

        # Plot BPASS solution
        if not show_bpass:
            continue

        y1 = src.tab_sed[np.argmin(np.abs(src.tab_waves_c - wave)),:]

    ##
    # Plot spectra
    ##
    for i, _t in enumerate([1, 10, 100]):

        y2 = toy.tab_sed[:,np.argmin(np.abs(toy.tab_t - _t))]

        # Plot BPASS solution
        if not show_bpass:
            continue

        y1 = src.tab_sed[:,np.argmin(np.abs(src.tab_t - _t))]

    ##
    # Make sure the spectra we put in are the spectra we get out.
    # e.g., do we recover UV slope of -2 if that's what we put in?
    ##

    beta = pop1.Beta(6., rest_wave=(1600., 2300.), dlam=10)
    mags = pop1.get_mags(6.)

    ok = beta != -99999

    assert np.allclose(beta[ok==1], pop1.src.pf['source_toysps_beta']), \
        "Not recovering beta={}! Mean={}".format(
            pop1.src.pf['source_toysps_beta'], beta[ok==1].mean())

    ##
    # Test adaptive time-stepping in spectral synthesis.
    ##

    tarr1 = np.arange(0, 1000, 1.)
    tarr2 = np.arange(0, 1000, dt_coarse)
    sfh1 = np.ones_like(tarr1)
    sfh2 = np.ones_like(tarr2)

    ss = ares.core.SpectralSynthesis()
    ss.src = toy

    ss2 = ares.core.SpectralSynthesis()
    ss2.src = toy
    ss2.oversampling_enabled = False
    ss2.oversampling_below = oversample_age

    t1 = time.time()
    L1 = ss.get_lum(sfh=sfh1, tarr=tarr1, load=False)
    t2 = time.time()

    print('dt=1', t2 - t1)

    t1 = time.time()
    L2 = ss.get_lum(sfh=sfh2, tarr=tarr2, load=False)
    t2 = time.time()
    print('dt={}, oversampling ON:'.format(dt_coarse), t2 - t1)

    t1 = time.time()
    L3 = ss2.get_lum(sfh=sfh2, tarr=tarr2, load=False)
    t2 = time.time()
    print('dt=10, oversampling OFF:', t2 - t1)

    def staircase(x, dx=10):

        N = x.size
        y = np.zeros_like(x)
        M = dx

        assert N % dx == 0

        ct = 0
        for i, xx in enumerate(x):

            y[M*ct:M*(ct+1)] = ct + 1

            if i % M == 0:
                ct += 1

        y[M*ct:M*(ct+1)] = ct + 1

        return y

    ##
    # Test with 'staircase' SFH.
    ##
    sfh1 = staircase(tarr1, dx=100)
    sfh2 = staircase(tarr2, dx=100//dt_coarse)

    L1 = ss.get_lum(sfh=sfh1, tarr=tarr1)
    L2 = ss.get_lum(sfh=sfh2, tarr=tarr2)
    L3 = ss2.get_lum(sfh=sfh2, tarr=tarr2)

    # Check validity of over-sampling for non-constant SFH
    # Just take mean error over long time as the solutions will differ
    # slightly (10%ish) at sharp discontinuities in SFH.
    err = []
    for j, _t in enumerate(tarr2):
        if _t == 0:
            continue

        i = np.argmin(np.abs(_t - tarr1))

        err.append(np.abs(L2[j] - L1[i]) / L1[i])


    assert np.mean(err) < 0.01

    ##
    # Test batch mode
    ##

    sfh2 = np.array([sfh2] * 10)

    L2b = ss.get_lum(sfh=sfh2, tarr=tarr2)
    L3b = ss2.get_lum(sfh=sfh2, tarr=tarr2)

    assert np.all(L2b[0] == L2)
    assert np.all(L3b[0] == L3)

        #print("Mean error in L(t) with oversampling at t<{} Myr: {}".format(oversample_age,
        #    np.mean(err)))

if __name__ == '__main__':
    test()
