"""

test_physics_collisional_coupling.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri Nov  9 16:01:47 2012

Description:

"""

import numpy as np
from ares.physics import Hydrogen

def test():

    hydr = Hydrogen(interp_cc='cubic')
    hydr2 = Hydrogen(interp_cc='linear')

    # Relevant temperature range + a little bit to make sure our interpolation
    # bounds are obeyed.
    T = np.logspace(-0.5, 4.5, 500)

    ok = 1
    for suffix in ['H', 'e']:

        tab = hydr.tabulated_coeff['kappa_{!s}'.format(suffix)]

        if suffix == 'H':
            interp = hydr.kappa_H(hydr.tabulated_coeff['T_{!s}'.format(suffix)])
        else:
            interp = hydr.kappa_e(hydr.tabulated_coeff['T_{!s}'.format(suffix)])

        # Numbers small so ignore absolute tolerance
        ok *= np.allclose(tab, interp, atol=0.0)

    assert ok, "Error in computation of coupling coefficients."

    # Also check extrapolation
    # If off, should just use last value in table
    Tlo = hydr.tabulated_coeff['T_H'].min()
    assert hydr.kappa_H(Tlo) == hydr.kappa_H(Tlo / 2.)
    assert hydr.kappa_e(Tlo) == hydr.kappa_e(Tlo / 2.)

    hydr_e = Hydrogen(extrapolate_coupling=True)
    assert hydr_e.kappa_H(Tlo) > hydr_e.kappa_H(Tlo / 2.)
    assert hydr_e.kappa_H(Tlo) > hydr_e.kappa_H(Tlo / 2.)

if __name__ == '__main__':
    test()
