"""

test_populations_ensemble.py

Author: Jordan Mirocha
Affiliation: McGill
Created on: Fri 27 Mar 2020 10:16:25 EDT

Description:

"""

import ares
import numpy as np
from ares.physics.Constants import rhodot_cgs

def test():
    pars = ares.util.ParameterBundle('mirocha2020:univ')
    pars.update(ares.util.ParameterBundle('testing:galaxies'))

    # Can't actually do this test yet because we don't have access to
    # even time-spaced HMFs/HGHs on travis.ci

    pop = ares.populations.GalaxyPopulation(**pars)

    # Test I/O. Should add more here eventually.
    pop.save('test_ensemble', clobber=True)

    # Test SFRD
    sfrd = pop.SFRD(6.) * rhodot_cgs

    assert 1e-3 <= sfrd <= 1, "SFRD unreasonable"

    csfrd = pop.cSFRD(6., Mh=1e10)

    assert csfrd < 1., "cSFRD >= SFRD!?"

    logMh, logf_stell, std = pop.get_smhm(6.)
    assert np.all(logf_stell < 1)

    # Test UVLF
    mags = np.arange(-30, 10, 0.1)
    mags_cr = np.arange(-30, 10, 1.)
    x, phi = pop.get_lf(6., mags, absolute=True)
    x2, phi2 = pop.LuminosityFunction(6., mags, absolute=True) # backward compat
    assert np.allclose(phi, phi2)

    ok = np.isfinite(phi)
    assert 1e-4 <= np.interp(-18, mags, phi) <= 1e-1, "UVLF unreasonable!"

    x, phi_c = pop.get_lf(6., mags, absolute=True)

    assert np.array_equal(phi[ok==1], phi_c[ok==1]), "UVLF cache not working!"

    # Test stellar mass function
    log10Ms = np.arange(6, 13, 0.5)
    phi = pop.get_smf(6., log10Ms)
    ok = np.isfinite(phi)

    assert 1e-4 <= np.interp(9, log10Ms, phi) <= 1e-1, "GSMF unreasonable!"

    phi_c = pop.get_smf(6., log10Ms)
    assert np.array_equal(phi[ok==1], phi_c[ok==1]), "GSMF cache not working!"

    # Just check dust masss etc.
    Md = pop.get_field(6., 'Md')

    assert 1e4 <= np.mean(Md) <= 1e10, "Dust masses unreasonable!"

    # Test extinction
    AUV = pop.get_AUV(6., magbins=mags_cr, return_binned=False)

    assert np.all(AUV >= 0), "AUV < 0! AUV={}".format(AUV)
    assert 0 < np.mean(AUV) <= 3, "AUV unreasonable!"

    AUV2 = pop.get_AUV(6., magbins=mags_cr, return_binned=True)
    assert AUV2.size == mags_cr.size
    AUV20 = AUV2[np.argmin(np.abs(mags_cr + 20))]
    AUV16 = AUV2[np.argmin(np.abs(mags_cr + 16))]
    assert AUV20 > AUV16, \
        "AUV should increase as MUV becomes more negative! {} {}".format(AUV20, AUV16)

    # Test AUV(Mstell)
    AUV3 = pop.get_AUV(6., magbins=mags_cr, return_binned=True, Mstell=1e10,
        massbins=np.arange(6, 12, 0.5))
    assert AUV3 > 0.0

    # Test UV slope
    b_hst = pop.get_uv_slope(6., presets='hst', dlam=100.)
    assert -3 <= np.nanmean(b_hst) <= -1, \
        "UV slopes unreasonable! Beta={}".format(b_hst)

    b_hst_b = pop.get_uv_slope(6., presets='hst', dlam=100.,
        return_binned=True, Mbins=mags_cr)

    filt, mag_hst = pop.get_mags(6., presets='hst', method='gmean', dlam=100.)

    b20 = b_hst_b[np.argmin(np.abs(mags_cr + 20))]
    b16 = b_hst_b[np.argmin(np.abs(mags_cr + 16))]
    assert b20 > b16, \
        "Beta should increase as MUV becomes more negative! {} {}".format(b20, b16)

    dBdMUV, func1, func2 = pop.get_dBeta_dMUV(6., mags_cr, presets='hst', dlam=100.,
        return_funcs=True, model='exp')
    assert np.all(dBdMUV < 0)

    # Simple LAE model
    x, xLAE, std = pop.get_lae_fraction(6, bins=mags_cr)
    assert np.mean(np.diff(xLAE) / np.diff(x))

    # Get single halo history
    hist = pop.get_history(20)

    # Get z=0 spectra
    spec = pop.get_spec_obs(6., waves=np.array([1600]))

    # Test galaxy bias calculation
    b = pop.get_bias(6., limit=-19.4, absolute=True, wave=1600.)
    assert 4 <= b <= 6, "bias unreasonable! b={}".format(b)

    b = pop.get_bias(6., limit=28, absolute=False, wave=1600.)
    assert 2 <= b <= 10, "bias unreasonable! b={}".format(b)

    b = pop.get_bias(6., limit=1e10, cut_in_mass=True, wave=1600.)
    assert 3 <= b <= 5, "bias unreasonable! b={}".format(b)

    # Surface density
    amag_bins = np.arange(20, 45, 0.1)
    x, Sigma = pop.get_surface_density(6, bins=amag_bins)
    assert 1e3 <= Sigma[np.argmin(np.abs(amag_bins - 27))] <= 1e4

    # Test nebular line emission stuff
    pars['pop_nebular'] = 2
    pop_neb = ares.populations.GalaxyPopulation(**pars)

    owaves, f_lya = pop_neb.get_line_flux(6, 'Ly-a')

    assert 1e-20 <= np.mean(f_lya) <= 1e-16, "Ly-a fluxes unreasonable!"

    # Test routines to retrieve MUV-Beta, AUV, etc.
    AUV = pop.get_AUV(6.)

if __name__ == '__main__':
    test()
