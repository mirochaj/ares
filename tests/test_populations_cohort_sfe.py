"""

test_pq_fstar.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Wed Jun 15 15:47:03 PDT 2016

Description:

"""

import ares
import numpy as np

pars_pl = \
{
'pop_fstar': 'pq',
'pq_func': 'pl',
'pq_func_var': 'Mh',
'pq_func_par0': 1e-1,
'pq_func_par1': 1e11,
'pq_func_par2': 0.6,
'pq_func_par3': 0.,
'pq_val_ceil': 0.1,
}

pars_dpl = \
{
'pop_fstar': 'pq',
'pq_func': 'dpl',
'pq_func_var': 'Mh',
'pq_func_par0': 1e-1,
'pq_func_par1': 1e11,
'pq_func_par2': 0.6,
'pq_func_par3': 0.,
'pq_func_par4': 1e14,      # Normalization mass
}

pars_pwpl = \
{
'pop_fstar': 'pq',
'pq_func': 'pwpl',
'pq_func_var': 'Mh',
'pq_func_par0': 1e-1,
'pq_func_par1': 0.6,
'pq_func_par2': 1e-1,
'pq_func_par3': 0.,
'pq_func_par4': 1e11,
}

pars_ramp = \
{
'pop_fstar': 'pq',
'pq_func': 'logramp',
'pq_func_var': 'Mh',
'pq_func_par0': 1e-3,
'pq_func_par1': 9,
'pq_func_par2': 1e-1,
'pq_func_par3': 11,
}

# Next two have evolving pieces
pars_pl_w_zdep = \
{
'pop_fstar': 'pq[0]',
'pq_func[0]': 'pl_evolN',
'pq_func_var[0]': 'Mh',
'pq_func_var2[0]': '1+z',
'pq_func_par0[0]': 1e-2,
'pq_func_par1[0]': 1e10,
'pq_func_par2[0]': 0.6,
'pq_func_par3[0]': 7.,
'pq_func_par4[0]': 1.,
'pq_val_ceil[0]': 0.1,
}

pars_dpl_Mofz = \
{
'pop_fstar': 'pq[0]',
'pq_func[0]': 'dpl_evolP',
'pq_func_var[0]': 'Mh',
'pq_func_var2[0]': '1+z',

'pq_func_par0[0]': 1e-2,
'pq_func_par1[0]': 1e12,
'pq_func_par2[0]': 0.6,
'pq_func_par3[0]': -0.5,
'pq_func_par4[0]': 1e10,
'pq_func_par5[0]': 7.,
'pq_func_par6[0]': -1.,
}

pars_dpl_Nofz = \
{
'pop_fstar': 'pq[0]',
'pq_func[0]': 'dpl_evolN',
'pq_func_var[0]': 'Mh',
'pq_func_var2[0]': '1+z',

'pq_func_par0[0]': 1e-2,
'pq_func_par1[0]': 1e12,
'pq_func_par2[0]': 0.6,
'pq_func_par3[0]': -0.5,
'pq_func_par4[0]': 1e10,
'pq_func_par5[0]': 7.,
'pq_func_par6[0]': 1.,
}

def test():

    Mh = np.logspace(7, 15, 200)

    ls = '-', '--', ':', '-.'
    lw = 2, 2, 4, 4
    labels = ['pl_w_ceil', 'dpl', 'pwpl', 'ramp']
    for i, pars in enumerate([pars_pl, pars_dpl, pars_pwpl, pars_ramp]):
        pop = ares.populations.GalaxyPopulation(pop_sfr_model='sfe-func', **pars)

        fnow = pop.get_sfe(z=6., Mh=Mh).copy()

        if (i > 0) and (labels[i] != 'ramp'):
            assert np.allclose(fnow[Mh <= 1e8], fprev[Mh <= 1e8], rtol=5e-2)
            assert np.allclose(fnow[Mh >= 1e14], fprev[Mh >= 1e14], rtol=5e-2)

        fprev = fnow.copy()

    pop1 = ares.populations.GalaxyPopulation(pop_sfr_model='sfe-func',
        **pars_pl_w_zdep)
    pop2 = ares.populations.GalaxyPopulation(pop_sfr_model='sfe-func',
        **pars_dpl_Mofz)
    pop3 = ares.populations.GalaxyPopulation(pop_sfr_model='sfe-func',
        **pars_dpl_Nofz)

    all_sfe = []
    for j, pop in enumerate([pop1, pop2, pop3]):
        for i, z in enumerate([6, 10]):
            sfe = pop.get_sfe(z=z, Mh=Mh)
            all_sfe.append(sfe)

    assert ~np.all(all_sfe[0] == all_sfe[1]), "No SFE evolution detected!"
    assert ~np.all(all_sfe[2] == all_sfe[3]), "No SFE evolution detected!"
    assert ~np.all(all_sfe[4] == all_sfe[5]), "No SFE evolution detected!"
    assert np.allclose(all_sfe[0][Mh <= 1e10], all_sfe[2][Mh <= 1e10],
        rtol=1e-2, atol=0), "Mismatch at low Mh!"
    assert np.allclose(all_sfe[0][Mh <= 1e10], all_sfe[4][Mh <= 1e10],
        rtol=1e-2, atol=0), "Mismatch at low Mh!"


    # Test abundance matching
    pars = ares.util.ParameterBundle('mirocha2017:base').pars_by_pop(0,1)
    pars.update(ares.util.ParameterBundle('testing:galaxies'))
    pop = ares.populations.GalaxyPopulation(**pars)

    bins = np.arange(-25, 0, 0.1)
    def uvlf(z, MUV):
        mags, phi = pop.get_uvlf(z, bins)
        return np.interp(MUV, mags, phi)

    pars2 = pars.copy()
    pars2['pop_sfr_model'] = 'uvlf'
    pars2['pop_uvlf'] = uvlf

    pop_ham = ares.populations.GalaxyPopulation(**pars2)

    fstar1 = pop.get_sfe(z=6, Mh=Mh)
    fstar1b = pop.get_fstar(z=6, Mh=Mh)

    assert np.allclose(fstar1, fstar1b)

    fstar2 = pop_ham.run_abundance_match(6, Mh)
    fstar2b = pop_ham.get_sfe(z=6, Mh=Mh)

    ok = np.logical_and(Mh >= 1e9, Mh <= 1e13)

    import matplotlib.pyplot as pl
    pl.loglog(Mh, fstar1, ls='-')
    pl.loglog(Mh, fstar2, ls='--')
    input('<enter>')

    assert np.allclose(fstar1[ok==1], fstar2[ok==1], rtol=1e-1)

    # Check tabulated fstar (slow)
    #fstar2c = pop_ham.tab_fstar[np.argmin(np.abs(6 - pop_ham.halos.tab_z))]

    # Check 21cmFAST parameterization
    pars_cmfast = ares.util.ParameterBundle('park2019:base').pars_by_pop(0, 1)
    pop_cmfast = ares.populations.GalaxyPopulation(**pars_cmfast)

    x, phi = pop_cmfast.get_uvlf(6, bins)

if __name__ == '__main__':
    test()
