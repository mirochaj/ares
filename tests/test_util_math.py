"""

test_util_stats.py

Author: Jordan Mirocha
Affiliation: McGill
Created on: Tue 24 Mar 2020 22:11:31 EDT

Description:

"""

import numpy as np
from scipy.interpolate import interp1d
from ares.util.Math import interp1d_wrapper, forward_difference, \
    central_difference, five_pt_stencil, LinearNDInterpolator, smooth

def test():

    # First, test my dumb wrapper around interp1d
    x = np.linspace(0, 4 * np.pi, 100)
    y = np.sin(x)

    func1 = interp1d(x, y, kind='cubic')
    func2 = interp1d_wrapper(x, y, kind='cubic')
    func3 = LinearNDInterpolator(x, y)

    x2 = np.linspace(0, 4 * np.pi, 50)

    f1 = func1(x2)
    f2 = func2(x2)
    f3 = func3(x2)

    assert np.array_equal(f1, f2)

    # Test derivative routines
    x1, dydx1 = forward_difference(x, y)
    x2, dydx2 = central_difference(x, y)
    x3, dydx3 = five_pt_stencil(x, y)

    # Smoothing
    d = y + np.random.normal(scale=0.5, size=y.size)
    std = np.std(d - y)
    ds_b = smooth(d, 5, kernel='boxcar')
    ds_g = smooth(d, 5, kernel='gaussian')

    assert np.std(ds_b - y) < std
    assert np.std(ds_g - y) < std

    # Next, test LinearNDInterpolator
    _x = _y = np.linspace(0, 5, 100)
    xx, yy = np.meshgrid(_x, _y)
    f = np.sin(xx) + np.cos(yy)

    func2d = LinearNDInterpolator([_x, _y], f)
    f0 = func2d(np.array([0.5, 1.3]))

    _x = _y = _z = np.linspace(0, 5, 100)
    xx, yy, zz = np.meshgrid(_x, _y, _z)
    g = np.sin(xx) + np.cos(yy) + + np.tan(zz)

    func3d = LinearNDInterpolator([_x, _y, _z], g)
    g0 = func3d(np.array([0.5, 1.3, 1.5]))

if __name__ == '__main__':
    test()
