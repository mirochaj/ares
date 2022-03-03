"""

test_physics_HI_lya.py

Author: Jordan Mirocha
Affiliation: McGill University
Created on: Mon 14 Feb 2022 15:24:50 EST

Description:

"""

import numpy as np
from ares.physics import Hydrogen
from ares.simulations import Global21cm

def test():

    Tarr = np.logspace(-1, 2)

    hydr = Hydrogen(approx_Salpha=0)

    z = 22.
    Tk = 10.
    xHII = 2.19e-4
    x = np.arange(-100, 100)

    Jc = np.array([hydr.get_lya_profile(z, Tk, xx, continuum=1, xHII=xHII) \
        for xx in x])
    Ji = np.array([hydr.get_lya_profile(z, Tk, xx, continuum=0, xHII=xHII) \
        for xx in x])

    Ji_norm = Ji * Jc[x==0]
    Ji_norm[x < 0] = Jc[x < 0]

    Ic = hydr.get_lya_EW(z, Tk, continuum=1)
    Ii = hydr.get_lya_EW(z, Tk, continuum=0)

    # Compare to Mittal & Kulkarni (2019) who actually quote numbers :)
    # Be lenient since we have different cosmologies.
    assert abs(Ic - 20.11) < 1
    assert abs(Ii - -5.75) < 1

    heat = hydr.get_lya_heating(z, Tk, Jc=1e-12, Ji=1e-13)

    # Compare to FP06 approximation, eventually Chuzhoy & Shapiro
    hydr_fp06 = Hydrogen(approx_Salpha=3)
    hydr_mk18 = Hydrogen(approx_Salpha=5)

    heat_fp06 = hydr_fp06.get_lya_heating(z, Tk, Jc=1e-12, Ji=1e-13)
    heat_mk18 = hydr_fp06.get_lya_heating(z, Tk, Jc=1e-12, Ji=1e-13)

    assert abs(heat_fp06 - heat) / heat < 0.1
    assert abs(heat_mk18 - heat) / heat < 0.1

    Ii_fp06 = hydr_fp06.get_lya_EW(z, Tk, continuum=0)
    Ii_mk18 = hydr_mk18.get_lya_EW(z, Tk, continuum=0)

    assert abs(Ii - Ii_fp06) / Ii < 0.1
    assert abs(Ii - Ii_mk18) / Ii < 0.1

if __name__ == '__main__':
    test()
