"""

test_phenom_pq.py

Author: Jordan Mirocha
Affiliation: McGill
Created on: Sat 28 Mar 2020 15:49:13 EDT

Description:

"""

import ares
import numpy as np
from ares.physics.Constants import rhodot_cgs

def test(atol=1e-4):

    # Test all parameterized quantities through SFRD.

    pars = {'pop_sfr_model': 'sfrd-func'}

    # Power-law SFRD
    pars['pop_sfrd'] = 'pq[0]'
    pars['pq_func[0]'] = 'pl'
    pars['pq_func_var[0]'] = '1+z'
    pars['pq_func_par0[0]'] = 1e-2 / rhodot_cgs
    pars['pq_func_par1[0]'] = 10.
    pars['pq_func_par2[0]'] = -2.
    pop = ares.populations.GalaxyPopulation(**pars)

    assert pop.get_sfrd(z=9.) * rhodot_cgs == 1e-2, \
        "Problem with PL SFRD"

    # Power-law SFRD with evolving normalization
    pars['pop_sfrd'] = 'pq[0]'
    pars['pq_func[0]'] = 'pl_evolN'
    pars['pq_func_var[0]'] = '1+z'
    pars['pq_func_var2[0]'] = '1+z'
    pars['pq_func_par0[0]'] = 1e-2 / rhodot_cgs
    pars['pq_func_par1[0]'] = 10.
    pars['pq_func_par2[0]'] = -2.
    pars['pq_func_par3[0]'] = 10.
    pars['pq_func_par4[0]'] = 0.
    pop = ares.populations.GalaxyPopulation(**pars)

    assert pop.get_sfrd(z=9.) * rhodot_cgs == 1e-2, \
        "Problem with PL (evolving norm) SFRD"

    # Exponential SFRD: p0 * e^{(x / p1)^p2}
    pars['pop_sfrd'] = 'pq[0]'
    pars['pq_func[0]'] = 'exp'
    pars['pq_func_var[0]'] = '1+z'
    pars['pq_func_par0[0]'] = 1e-2 / rhodot_cgs
    pars['pq_func_par1[0]'] = 10.
    pars['pq_func_par2[0]'] = 1.
    pop = ares.populations.GalaxyPopulation(**pars)

    assert pop.get_sfrd(z=9.) * rhodot_cgs / np.exp(1.) == 1e-2, \
        "Problem with exp SFRD"

    # Exponential SFRD: p0 * e^{(x / p1)^p2}
    pars['pop_sfrd'] = 'pq[0]'
    pars['pq_func[0]'] = 'exp-'
    pars['pq_func_var[0]'] = '1+z'
    pars['pq_func_par0[0]'] = 1e-2 / rhodot_cgs
    pars['pq_func_par1[0]'] = 10.
    pars['pq_func_par2[0]'] = 1.
    pop = ares.populations.GalaxyPopulation(**pars)

    sfrd = pop.get_sfrd(z=9.) * rhodot_cgs / np.exp(-1.)
    assert abs(sfrd - 1e-2) < atol, \
        "Problem with exp- SFRD"

    # Gaussian SFRD
    pars['pop_sfrd'] = 'pq[0]'
    pars['pq_func[0]'] = 'normal'
    pars['pq_func_var[0]'] = '1+z'
    pars['pq_func_par0[0]'] = 1e-2 / rhodot_cgs
    pars['pq_func_par1[0]'] = 10.
    pars['pq_func_par2[0]'] = 1.
    pop = ares.populations.GalaxyPopulation(**pars)

    assert pop.get_sfrd(z=9.) * rhodot_cgs == 1e-2, \
        "Problem with normal SFRD"

    # Log-normal SFRD
    pars['pop_sfrd'] = 'pq[0]'
    pars['pq_func[0]'] = 'lognormal'
    pars['pq_func_var[0]'] = '1+z'
    pars['pq_func_par0[0]'] = 1e-2 / rhodot_cgs
    pars['pq_func_par1[0]'] = 1.
    pars['pq_func_par2[0]'] = 1.
    pop = ares.populations.GalaxyPopulation(**pars)

    assert pop.get_sfrd(z=9.) * rhodot_cgs == 1e-2, \
        "Problem with log-normal SFRD"

    # Log-normal SFRD
    pars['pop_sfrd'] = 'pq[0]'
    pars['pq_func[0]'] = 'pwpl'
    pars['pq_func_var[0]'] = '1+z'
    pars['pq_func_par0[0]'] = 1e-2 / rhodot_cgs
    pars['pq_func_par1[0]'] = 1.
    pars['pq_func_par2[0]'] = 1e-2 / rhodot_cgs
    pars['pq_func_par3[0]'] = 1.
    pars['pq_func_par4[0]'] = 10. # Behavior different above and below 1+z=20
    pop = ares.populations.GalaxyPopulation(**pars)

    sfrd = pop.get_sfrd(z=9.) * rhodot_cgs
    assert abs(sfrd - 1e-2) < atol, \
        "Problem with piece-wise power-law SFRD"

    # Ramp SFRD
    pars['pop_sfrd'] = 'pq[0]'
    pars['pq_func[0]'] = 'ramp'
    pars['pq_func_var[0]'] = '1+z'
    pars['pq_func_par0[0]'] = 1e-2 / rhodot_cgs
    pars['pq_func_par1[0]'] = 15.
    pars['pq_func_par2[0]'] = 1e-3
    pars['pq_func_par3[0]'] = 30.
    pop = ares.populations.GalaxyPopulation(**pars)

    sfrd = pop.get_sfrd(z=9.) * rhodot_cgs
    assert abs(sfrd - 1e-2) < atol, \
        "Problem with 'ramp' SFRD"

    # Log-ramp SFRD
    pars['pop_sfrd'] = 'pq[0]'
    pars['pq_func[0]'] = 'logramp'
    pars['pq_func_var[0]'] = '1+z'
    pars['pq_func_par0[0]'] = 1e-2 / rhodot_cgs
    pars['pq_func_par1[0]'] = np.log10(15.)
    pars['pq_func_par2[0]'] = 1e-3
    pars['pq_func_par3[0]'] = np.log10(30.)
    pop = ares.populations.GalaxyPopulation(**pars)

    sfrd = pop.get_sfrd(z=9.) * rhodot_cgs
    assert abs(sfrd - 1e-2) < atol, \
        "Problem with 'logramp' SFRD"

    # A few different tanh functions
    pars['pop_sfrd'] = 'pq[0]'
    pars['pq_func[0]'] = 'tanh_abs'
    pars['pq_func_var[0]'] = '1+z'
    pars['pq_func_par0[0]'] = 1e-2 / rhodot_cgs
    pars['pq_func_par1[0]'] = 1e-3 / rhodot_cgs
    pars['pq_func_par2[0]'] = 20.
    pars['pq_func_par3[0]'] = 0.5
    pop = ares.populations.GalaxyPopulation(**pars)

    sfrd = pop.get_sfrd(z=9.) * rhodot_cgs
    assert abs(sfrd - 1e-2) < atol, \
        "Problem with 'tanh_abs' SFRD"

    pars['pop_sfrd'] = 'pq[0]'
    pars['pq_func[0]'] = 'tanh_rel'
    pars['pq_func_var[0]'] = '1+z'
    pars['pq_func_par0[0]'] = 1. / rhodot_cgs
    pars['pq_func_par1[0]'] = 1e-2 / rhodot_cgs
    pars['pq_func_par2[0]'] = 20.
    pars['pq_func_par3[0]'] = 0.5
    pop = ares.populations.GalaxyPopulation(**pars)

    sfrd = pop.get_sfrd(z=9.) * rhodot_cgs
    assert abs(sfrd - 1e-2) < atol, \
        f"Problem with 'tanh_rel' SFRD (={sfrd})"

    pars['pop_sfrd'] = 'pq[0]'
    pars['pq_func[0]'] = 'logtanh_abs'
    pars['pq_func_var[0]'] = '1+z'
    pars['pq_func_par0[0]'] = 1e-2 / rhodot_cgs
    pars['pq_func_par1[0]'] = 1e-3 / rhodot_cgs
    pars['pq_func_par2[0]'] = np.log10(20.)
    pars['pq_func_par3[0]'] = 0.05
    pop = ares.populations.GalaxyPopulation(**pars)

    sfrd = pop.get_sfrd(z=9.) * rhodot_cgs
    assert abs(sfrd - 1e-2) < atol, \
        "Problem with 'logtanh_abs' SFRD"

    pars['pop_sfrd'] = 'pq[0]'
    pars['pq_func[0]'] = 'logtanh_rel'
    pars['pq_func_var[0]'] = '1+z'
    pars['pq_func_par0[0]'] = 1. / rhodot_cgs
    pars['pq_func_par1[0]'] = 1e-2 / rhodot_cgs
    pars['pq_func_par2[0]'] = np.log10(20.)
    pars['pq_func_par3[0]'] = 0.05
    pop = ares.populations.GalaxyPopulation(**pars)

    sfrd = pop.get_sfrd(z=9.) * rhodot_cgs
    assert abs(sfrd - 1e-2) < atol, \
        f"Problem with 'tanh_rel' SFRD (={sfrd})"

    # A few step functions
    pars['pop_sfrd'] = 'pq[0]'
    pars['pq_func[0]'] = 'step_rel'
    pars['pq_func_var[0]'] = '1+z'
    pars['pq_func_par0[0]'] = 1e-1 / rhodot_cgs
    pars['pq_func_par1[0]'] = 1e-1
    pars['pq_func_par2[0]'] = 20.
    pop = ares.populations.GalaxyPopulation(**pars)

    sfrd = pop.get_sfrd(z=9.) * rhodot_cgs
    assert abs(sfrd - 1e-2) < atol, \
        "Problem with 'step_rel' SFRD"

    pars['pop_sfrd'] = 'pq[0]'
    pars['pq_func[0]'] = 'step_abs'
    pars['pq_func_var[0]'] = '1+z'
    pars['pq_func_par0[0]'] = 1e-2 / rhodot_cgs
    pars['pq_func_par1[0]'] = 1e-3
    pars['pq_func_par2[0]'] = 20.
    pop = ares.populations.GalaxyPopulation(**pars)

    sfrd = pop.get_sfrd(z=9.) * rhodot_cgs
    assert abs(sfrd - 1e-2) < atol, \
        "Problem with 'step_abs' SFRD"

    # Next: various double power-laws


if __name__ == '__main__':
    test()
