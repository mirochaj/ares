"""

test_util_stats.py

Author: Jordan Mirocha
Affiliation: McGill
Created on: Tue 24 Mar 2020 22:11:31 EDT

Description: 

"""

import ares
import numpy as np

def test(tol=1e-2):
    
    # Make unit Gaussian, test that we recover 68% confidence region
    x = np.arange(-5, 5, 0.001)
    y = ares.util.Stats.Gauss1D(x, pars=[0, 1., 0., 1.])
    assert y.max() == 1
    
    mu, (lo, hi) = ares.util.Stats.error_1D(x, y)
    assert abs(mu) < tol
    assert abs(1. - lo) < tol
    assert abs(1. - hi) < tol

    # Do the same thing for 2-D Gaussian
    # use GaussND, error_2D


    # Make some bins
    xe = np.arange(0, 1, 0.05)
    xc = ares.util.Stats.bin_e2c(xe)
    assert np.allclose(xc, xe[0:-1] + 0.025, rtol=tol)
    assert np.allclose(xe, ares.util.Stats.bin_c2e(xc))
    
    # Make some fake data to test scatter analysis on.
    y = x + 2 + np.random.normal(scale=0.3, size=x.size)
    xb = x[0:-1:100]
    _xb, yb, s, N = ares.util.Stats.quantify_scatter(x, y, xb)
    
    slope = np.mean(np.diff(yb) / np.diff(xb))
    
    assert abs(np.mean(s) - 0.3) < tol, "Something wrong with scatter."
    assert abs(slope - 1.) < tol, "Not recovering mean slope."
    

if __name__ == '__main__':
    test()
