"""

test_spec_synth_phot.py

Author: Jordan Mirocha
Affiliation: McGill
Created on: Mon  2 Dec 2019 10:32:47 EST

Description:

"""

import ares
import numpy as np
from ares.obs.Photometry import get_filters_from_waves
from ares.physics.Constants import flux_AB, cm_per_pc, s_per_myr

def test(tol=0.25):

    pars = ares.util.ParameterBundle('mirocha2020:univ')
    pars['pop_sed'] = 'sps-toy'
    # Turn off aging so we recover beta = -2
    pars["pop_toysps_alpha"] = 0.
    pars['pop_toysps_gamma'] = 0.
    pars['pop_dust_yield'] = 0
    pars['pop_dlam'] = 10.
    pars['pop_lmin'] = 1000.
    pars['pop_lmax'] = 3000.
    pars['pop_toysps_beta'] = -2.
    pars['pop_thin_hist'] = 0
    pars['pop_scatter_mar'] = 0
    pars['pop_Tmin'] = None # So we don't have to read in HMF table for Mmin
    pars['pop_Mmin'] = 1e8
    pars['pop_synth_minimal'] = False
    pars['pop_sed_degrade'] = None
    pars['tau_clumpy'] = None

    # Prevent use of hmf table
    tarr = np.arange(50, 2000, 1.)[-1::-1]
    cosm = ares.physics.Cosmology()
    zarr = cosm.z_of_t(tarr * s_per_myr)
    pars['pop_histories'] = {'t': tarr, 'z': zarr,
        'MAR': np.ones((1, tarr.size)), 'nh': np.ones((1, tarr.size)),
        'Mh': 1e10 * np.ones((1, tarr.size))}

    pop = ares.populations.GalaxyPopulation(**pars)

    _b14 = ares.data.read('bouwens2014')
    hst_shallow = _b14.filt_shallow
    hst_deep = _b14.filt_deep

    c94_windows = ares.data.read('calzetti1994').windows
    wave_lo = np.min(c94_windows)
    wave_hi = np.max(c94_windows)

    waves = np.arange(1000., 3000., 10.)
    load = False

    ##
    # Assert that magnitudes change with time, but that at fixed time snapshot,
    # different magnitude estimation techniques differ by < 0.2 mag.
    ##
    for i, z in enumerate([4.,5.,6.]):

        zstr = int(round(z))

        if zstr >= 7:
            filt_hst = hst_deep
        else:
            filt_hst = hst_shallow

        hist = pop.histories
        owaves, oflux = pop.synth.get_spec_obs(zobs=z, sfh=hist['SFR'],
            tarr=hist['t'], zarr=hist['z'], waves=waves, hist=hist,
            extras=pop.extras, load=load)

        # Compute observed magnitudes of all spectral channels
        dL = pop.cosm.LuminosityDistance(z)
        magcorr = 5. * (np.log10(dL / cm_per_pc) - 1.) - 2.5 * np.log10(1. + z)
        omags = -2.5 * np.log10(oflux / flux_AB) - magcorr

        mag_from_spec = omags[0,np.argmin(np.abs(1600. - waves))]

        # Compute observed magnitude at 1600A by hand from luminosity
        L = pop.get_lum(z, x=1600., units='Angstroms', load=load)
        f = L[0] * (1. + z) / (4. * np.pi * dL**2)
        mag_from_flux = -2.5 * np.log10(f / flux_AB) - magcorr

        # Use built-in method to obtain 1600A magnitude.
        mag_from_lum = pop.magsys.L_to_MAB(L[0])

        # Compute 1600A magnitude using different smoothing windows
        _f, mag_from_spec_20 = pop.get_mags(z, x=1600., window=21,
            load=load, units='Angstroms')
        _f, mag_from_spec_50 = pop.get_mags(z, x=1600., window=51,
            load=load, units='Angstroms')
        _f, mag_from_spec_100 = pop.get_mags(z, x=1600., window=201,
            load=load, units='Angstroms')

        # Different ways to estimate magnitude from HST photometry
        _f, mag_from_phot_mean = pop.get_mags(z, cam='hst',
            filters=filt_hst[zstr],
            method='gmean', load=load)
        _f, mag_from_phot_close = pop.get_mags(z, cam='hst',
            filters=filt_hst[zstr],
            method='closest', load=load, x=1600., units='Angstroms')
        _f, mag_from_phot_interp = pop.get_mags(z, cam='hst',
            filters=filt_hst[zstr],
            method='interp', load=load, x=1600., units='Angstroms')

        # These should be identical to machine precision
        assert abs(mag_from_spec-mag_from_flux) < 1e-8, \
            "These should all be identical! z={}".format(z)
        assert abs(mag_from_spec-mag_from_lum)  < 1e-8, \
            "These should all be identical! z={}".format(z)

        results = [mag_from_spec, mag_from_flux, mag_from_lum,
            mag_from_spec_20[0], mag_from_spec_50[0], mag_from_spec_100[0],
            mag_from_phot_mean[0], mag_from_phot_close[0], mag_from_phot_interp[0]]

        assert np.all(np.abs(np.diff(results)) < tol), \
            "Error in magnitudes! z={}".format(z)


if __name__ == '__main__':
    test()
