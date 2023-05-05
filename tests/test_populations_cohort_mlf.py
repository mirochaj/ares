"""

test_pq_mlf.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Fri May 20 15:38:19 PDT 2016

Description:

"""

import ares
import numpy as np

def test():

    # First, a population where we model the star formation efficiency as a PL
    pars_sfe = \
    {
    'pop_sfr_model': 'sfe-func',
    'pop_fstar': 'pq',
    'pop_mlf': None,

    # Put MAR in by hand?


    'pq_func': 'pl',
    'pq_func_var': 'Mh',
    'pq_func_par0': 1e-1,
    'pq_func_par1': 1e11,
    'pq_func_par2': 0.,
    }

    # Second, a population where we model the mass loading factor, and then
    # figure out the SFE. We should get the same answer!
    pars_mlf = \
    {
    'pop_sfr_model': 'sfe-func',
    'pop_fstar': None,
    'pop_mlf': 'pq',

    'pq_func': 'pl',
    'pq_func_var': 'Mh',
    'pq_func_par0': (1. / 1e-1) - 1.,  # Compute MLF from {SFE = 1 / (1  + MLF)}
    'pq_func_par1': 1e11,
    'pq_func_par2': 0.,
    }

    pop_sfe = ares.populations.GalaxyPopulation(**pars_sfe)
    pop_mlf = ares.populations.GalaxyPopulation(**pars_mlf)

    assert pop_sfe.get_sfe(Mh=1e10) == pop_mlf.get_sfe(Mh=1e10) == 1e-1, \
        "Mass evolution not working properly in both SFE and MLF approaches."

    # Third, a population where we parameterize the SFR function. Just take it
    # from one of the GalaxyCohort instances above.
    pars_sfr = \
    {
    'pop_fstar': None,
    'pop_mlf': None,
    'pop_sfr': 'pq',

    'pq_func': 'pl',
    'pq_func_var': 'Mh',
    'pq_func_par0': (1. / 1e-1) - 1.,  # Compute MLF from {SFE = 1 / (1  + MLF)}
    'pq_func_par1': 1e11,
    'pq_func_par2': 0.,
    }


    """
    Next: make sure redshift dependence works.
    """

    pars_sfe = \
    {
    'pop_sfr_model': 'sfe-func',
    'pop_fstar': 'pq',
    'pq_func': 'pl_evolN',
    'pq_func_var': 'Mh',
    'pq_func_var2': 'z',
    'pq_func_par0': 1e-1,
    'pq_func_par1': 1e11,
    'pq_func_par2': 0.6,
    'pq_func_par3': 7.,
    'pq_func_par4': 1.,

    }

    z1 = 10.
    z2 = 15.

    pop_sfe = ares.populations.GalaxyPopulation(**pars_sfe)

    correction = (z2 / z1)**pars_sfe['pq_func_par4']

    assert pop_sfe.get_sfe(z=z1, Mh=1e11) * correction == pop_sfe.get_sfe(z=z2, Mh=1e11), \
        "Redshift evolution not working properly for SFE."

    # Check some MAR corrections
    MAR = pop_sfe.halos.tab_MAR

    pop_eta_1 = ares.populations.GalaxyPopulation(pop_MAR_corr='integral',
        **pars_sfe)
    pop_eta_2 = ares.populations.GalaxyPopulation(pop_MAR_corr='slope',
        **pars_sfe)

    z = pop_sfe.halos.tab_z
    Mh = pop_sfe.halos.tab_M

    assert np.allclose(pop_sfe.dfcolldt(z), pop_eta_1.dfcolldt(z))
    assert np.allclose(pop_sfe.dfcolldt(z), pop_eta_2.dfcolldt(z))

    ok = np.logical_and(Mh >= 1e10, Mh <= 1e12)
    MAR = pop_sfe.get_mar(6, Mh=Mh)
    MAR1 = pop_eta_1.get_mar(6, Mh=Mh)
    MAR2 = pop_eta_2.get_mar(6, Mh=Mh)

    # These are effectively different models, so only looking for
    # OOM agreement.
    assert np.allclose(MAR[ok==1], MAR1[ok==1], rtol=1)
    assert np.allclose(MAR[ok==1], MAR2[ok==1], rtol=1)


if __name__ == '__main__':
    test()
