"""

test_populations_dust.py

Author: Jordan Mirocha
Affiliation: JPL / Caltech
Created on: Thu May 25 14:25:43 PDT 2023

Description:

"""

import ares
import numpy as np

def test():
    pars = ares.util.ParameterBundle('mirocha2023:base').pars_by_pop(0,1)
    pars.update(ares.util.ParameterBundle('testing:galaxies'))
    pars['pop_Z'] = (0.02, 0.02)
    pars['pop_age'] = (100, 100)
    pars['pop_ssp'] = False, False

    pop_Av = ares.populations.GalaxyPopulation(**pars)

    assert pop_Av.dust.get_transmission(1500, Av=0.5) \
         > pop_Av.dust.get_transmission(1500, Av=1)

    assert pop_Av.dust.get_transmission(2500, Av=0.5) \
         > pop_Av.dust.get_transmission(1500, Av=0.5)

    assert pop_Av.dust.get_attenuation(1500, Av=0.5) \
         < pop_Av.dust.get_attenuation(1500, Av=1)

    assert pop_Av.dust.get_attenuation(2500, Av=0.5) \
         < pop_Av.dust.get_attenuation(1500, Av=0.5)

    assert np.exp(-pop_Av.dust.get_opacity(1500, Av=0.5)) \
        == pop_Av.dust.get_transmission(1500, Av=0.5)

    pars2 = ares.util.ParameterBundle('mirocha2023:base').pars_by_pop(0,1)
    pars2.update(ares.util.ParameterBundle('testing:galaxies'))
    pars2['pop_Z'] = (0.02, 0.02)
    pars2['pop_age'] = (100, 100)
    pars2['pop_ssp'] = False, False
    pars2.update(ares.util.ParameterBundle('mirocha2020:dust_screen'))
    pars2['pop_dust_template'] = None
    pars2['pop_dust_absorption_coeff'] = 'pq[20]'
    pars2["pq_func[20]"] = 'pl'
    pars2['pq_func_var[20]'] = 'wave'
    pars2['pq_func_var_lim[20]'] = (0., np.inf)
    pars2['pq_func_var_fill[20]'] = 0.0
    pars2['pq_func_par0[20]'] = 1e5      # opacity at wavelength below
    pars2['pq_func_par1[20]'] = 1e3
    pars2['pq_func_par2[20]'] = -1.
    pop_Sd = ares.populations.GalaxyPopulation(**pars2)

    waves = np.linspace(1e3, 1e4, 100)
    assert np.all(pop_Sd.dust.get_transmission(waves, Sd=0.5e-5) <= 1)
    assert np.all(pop_Av.dust.get_transmission(waves, Av=0.5) <= 1)

    assert pop_Sd.dust.get_transmission(1500, Sd=0.5e-5) \
         > pop_Sd.dust.get_transmission(1500, Sd=1e-5)

    assert pop_Sd.dust.get_transmission(2500, Sd=0.5e-5) \
         > pop_Sd.dust.get_transmission(1500, Sd=0.5e-5)


    assert pop_Sd.dust.get_attenuation(1500, Sd=0.5e-5) \
         < pop_Sd.dust.get_attenuation(1500, Sd=1e-5)

    assert pop_Sd.dust.get_attenuation(2500, Sd=0.5e-5) \
         < pop_Sd.dust.get_attenuation(1500, Sd=0.5e-5)

    assert np.exp(-pop_Sd.dust.get_opacity(1500, Sd=0.5e-5)) \
        == pop_Sd.dust.get_transmission(1500, Sd=0.5e-5)


    # Check that all works if Mh is an array
    Sd = np.logspace(-6, -5, 10)
    tau = pop_Sd.dust.get_opacity(waves, Sd=Sd)

    Ms = np.logspace(7, 11, 100)

    assert tau.shape == (Sd.size, waves.size)

    pars0 = ares.util.ParameterBundle('mirocha2023:base').pars_by_pop(0,1)
    pars0.update(ares.util.ParameterBundle('testing:galaxies'))
    pars0['pop_Z'] = (0.02, 0.02)
    pars0['pop_age'] = (100, 100)
    pars0['pop_ssp'] = False, False
    pars0['pq_func_par0[4]'] = 0 # Turn dust off
    pop0 = ares.populations.GalaxyPopulation(**pars0)

    L0 = pop0.get_lum(z=6, x=1600, units='Angstroms')
    ow, spec0 = pop0.get_spec_obs(z=6, waves=waves)

    # Call stuff through Cohort, Ensemble.
    for pop in [pop_Av, pop_Sd]:
        assert pop.is_dusty

        if pop.pf['pop_Av'] is not None:
            Av = pop.get_Av(z=6, Ms=Ms)
            assert np.all(Av >= 0)
        else:
            Sd = pop.get_dust_surface_density(z=6, Mh=pop.halos.tab_M)
            assert np.all(Sd >= 0)

        # Check luminosity: make sure dusty less luminous than dust-less
        L1600 = pop.get_lum(z=6, x=1600, units='Angstroms')
        assert np.all(L1600 <= L0)

        # Check spec_obs
        ow, spec = pop.get_spec_obs(z=6, waves=waves)

        assert np.all(spec <= spec0)

    ##
    # Check MUV-Beta approach to reddening
    pars_leg = ares.util.ParameterBundle('mirocha2020:legacy')
    pars_leg.update(ares.util.ParameterBundle('testing:galaxies'))
    pop_leg = ares.populations.GalaxyPopulation(**pars_leg)
    mags = np.arange(-25, -10, 0.1)
    for muvbeta in [-2., 'bouwens2014']:
        # Use same parameters as no-dust case to ensure systematic effect
        pars_irxb = ares.util.ParameterBundle('mirocha2020:legacy')
        pars_irxb.update(ares.util.ParameterBundle('testing:galaxies'))
        pars_irxb['pop_muvbeta'] = muvbeta
        pars_irxb['pop_irxbeta'] = 'meurer1999'
        pop_irxb = ares.populations.GalaxyPopulation(**pars_irxb)

        AUV = pop_irxb.dust.get_attenuation(wave=1600, MUV=-20, z=6)
        assert 0 <= AUV <= 3, "AUV unreasonable!"

        lf_wd = pop_irxb.get_uvlf(6, mags)[1]
        lf_0d = pop_leg.get_uvlf(6, mags)[1]

        diff = (lf_0d - lf_wd)

        # Check that UVLF is suppressed when we include dust.
        assert np.all(diff <= 1), \
            "Issue with phenomenological dust correction!"

if __name__ == '__main__':
    test()
