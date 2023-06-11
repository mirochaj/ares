"""

test_populations_ensemble.py

Author: Jordan Mirocha
Affiliation: McGill
Created on: Fri 27 Mar 2020 10:16:25 EDT

Description:

"""

import ares
import numpy as np
from ares.physics.Constants import rhodot_cgs, E_LL, cm_per_mpc, ev_per_hz

def test():
    pars = ares.util.ParameterBundle('mirocha2020:univ')
    pars.update(ares.util.ParameterBundle('testing:galaxies'))

    # Can't actually do this test yet because we don't have access to
    # even time-spaced HMFs/HGHs on travis.ci

    pop = ares.populations.GalaxyPopulation(**pars)

    # Test I/O. Should add more here eventually.
    pop.save('test_ensemble', clobber=True)

    z = pop.tab_z
    t = pop.tab_t

    # Test SFRD
    sfrd = pop.get_sfrd(6.) * rhodot_cgs

    assert 1e-3 <= sfrd <= 1, "SFRD unreasonable"

    sfrd10 = pop.get_sfrd_in_mass_range(6., Mlo=1e10)

    assert sfrd10 < sfrd, "sfrd(Mh>1e10) >= sfrd(all Mh)!?"

    logMh, logf_stell, std = pop.get_smhm(6.)
    assert np.all(logf_stell < 1)

    logMs, logSFR, err = pop.get_main_sequence(6.)

    # Test UVLF
    mags = np.arange(-30, 10, 0.1)
    mags_cr = np.arange(-30, 10, 1.)
    x, phi = pop.get_lf(6., mags, absolute=True)

    ok = np.isfinite(phi)
    assert 1e-4 <= np.interp(-18, mags, phi) <= 1e-1, "UVLF unreasonable!"

    x, phi_c = pop.get_lf(6., mags, absolute=True)

    assert np.array_equal(phi[ok==1], phi_c[ok==1]), "UVLF cache not working!"

    # SFR function
    x, phi = pop.get_sfr_df(6.)

    # MUV-Mstell
    MUV, log10Mst, err = pop.get_uvsm(6.)
    ok = np.isfinite(log10Mst)
    assert np.mean(np.diff(log10Mst[ok==1]) / np.diff(MUV[ok==1])) < 0

    # Test stellar mass function
    x, phi = pop.get_smf(6., Mbin=0.5)
    ok = np.isfinite(phi)

    assert 1e-4 <= np.interp(9, x, phi) <= 1e-1, "GSMF unreasonable!"

    x, phi_c = pop.get_smf(6., x)
    assert np.array_equal(phi[ok==1], phi_c[ok==1]), "GSMF cache not working!"

    # Just check dust masss etc.
    Md = pop.get_field(6., 'Md')

    assert 1e4 <= np.mean(Md) <= 1e10, "Dust masses unreasonable!"

    # Test extinction
    x, AUV = pop.get_AUV(6., magbins=mags_cr, return_binned=False)

    assert np.all(AUV >= 0), "AUV < 0! AUV={}".format(AUV)
    assert 0 < np.mean(AUV) <= 3, "AUV unreasonable!"

    x2, AUV2 = pop.get_AUV(6., magbins=mags_cr, return_binned=True)
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
    #b_hst = pop.get_uv_slope(6., presets='hst', dlam=100.)
    #assert -3 <= np.nanmean(b_hst) <= -1, \
    #    "UV slopes unreasonable! Beta={}".format(b_hst)

    #b_hst_b = pop.get_uv_slope(6., presets='hst', dlam=100.,
    #    return_binned=True, Mbins=mags_cr)

    #b20 = b_hst_b[np.argmin(np.abs(mags_cr + 20))]
    #b16 = b_hst_b[np.argmin(np.abs(mags_cr + 16))]
    #assert b20 > b16, \
    #    "Beta should increase as MUV becomes more negative! {} {}".format(b20, b16)

    filt, mag_hst = pop.get_mags(6., cam='hst', method='closest', x=1600,
        units='Angstroms', dlam=100.)


    #dBdMUV, func1, func2 = pop.get_dBeta_dMUV(6., mags_cr, presets='hst', dlam=100.,
    #    return_funcs=True, model='exp')
    #assert np.all(dBdMUV < 0)

    # Simple LAE model
    x, xLAE, std = pop.get_lae_fraction(6, bins=mags_cr)
    ok = np.isfinite(xLAE)

    # Just make sure x_LAE increases as MUV decreases. Note that we don't
    # have many galaxies in this model so just check that on avg the derivative
    # is positive in dxLAE/dMUV.
    assert np.mean(np.diff(xLAE[ok==1]) / np.diff(x[ok==1])) > 0

    # Get single halo history
    hist = pop.get_history(20)

    # Get z=0 spectra
    spec = pop.get_spec_obs(6., waves=np.array([1600]))

    # Test galaxy bias calculation
    b = pop.get_bias(6., limit=-19.4, absolute=True, x=1600.)
    assert 4 <= b <= 6, "bias unreasonable! b={}".format(b)

    b2 = pop.get_bias_from_scaling_relations(6.,
        smhm=(10**logMh, 10**logf_stell),  uvsm=(MUV, 10**log10Mst),
        limit=-19.4)
    b3 = pop.get_bias_from_scaling_relations(6.,
        smhm=(10**logMh, 10**logf_stell),  uvsm=(MUV, 10**log10Mst),
        limit=-19.4, use_dpl_smhm=True, Mpeak=1e12)

    # This is a bit hacky so the agreement shouldn't be great.
    assert abs(b2 - b) < 1
    assert abs(b3 - b) < 1

    b = pop.get_bias(6., limit=28, absolute=False, x=1600.)
    assert 2 <= b <= 10, "bias unreasonable! b={}".format(b)

    b = pop.get_bias(6., limit=1e10, cut_in_mass=True, x=1600.)
    assert 3 <= b <= 5, "bias unreasonable! b={}".format(b)

    # Surface density
    amag_bins = np.arange(20, 45, 0.1)
    x, Sigma = pop.get_surface_density(6, bins=amag_bins)
    sigma27 = Sigma[np.argmin(np.abs(amag_bins - 27))]
    assert 1e3 <= sigma27 <= 1e5, \
        f"Surface density ({sigma27} at m_AB=27) unreasonable!"

    # Test surface density integral sub-sampling. Should be a small effect.
    x1, Sigma1 = pop.get_surface_density(6, dz=0.1, bins=amag_bins)
    x2, Sigma2 = pop.get_surface_density(6, dz=0.1, bins=amag_bins,
        use_central_z=False, zstep=0.025)

    rdiff = np.abs(Sigma1 - Sigma2) / Sigma2
    assert rdiff[Sigma2 > 0].mean() < 0.15, \
        f"Evolution effect is too large! {rdiff[Sigma2 > 0].mean()}"

    # Try volume density
    x, n = pop.get_volume_density(6, bins=amag_bins)

    # Test nebular line emission stuff
    pars['pop_nebular'] = 2
    pop_neb = ares.populations.GalaxyPopulation(**pars)

    try:
        owaves, f_lya = pop_neb.get_line_flux(6, 'Ly-a')
        assert 1e-20 <= np.mean(f_lya) <= 1e-16, "Ly-a fluxes unreasonable!"
    except AssertionError:
        # Supposed to happen: in future would like to test line emission
        # but need finer SED table to do that, which doesn't ship with
        # lookup tables used for test suite.
        pass

    # Test routines to retrieve MUV-Beta, AUV, etc.
    AUV = pop.get_AUV(6.)

    # Emissivity stuff: just OOM check at the moment.
    zarr = np.arange(6, 30)
    e_ion = np.array([pop.get_emissivity(z, band=(E_LL, 1e2), units='eV') \
        for z in zarr]) * cm_per_mpc**3
    e_ion2 = np.array([pop.get_emissivity(z, band=(E_LL, 1e2), units='eV') \
        for z in zarr]) * cm_per_mpc**3     # check caching

    assert 1e37 <= np.mean(e_ion) <= 1e41
    assert np.allclose(e_ion, e_ion2)

    n_ion = np.array([pop.get_photon_density(z, band=(E_LL, 1e2), units='eV') \
        for z in zarr]) * cm_per_mpc**3

    assert 1e47 <= np.mean(n_ion) <= 1e51

if __name__ == '__main__':
    test()
